# utils.py
from __future__ import annotations

import numpy as np
import torch
import random
from typing import Dict, List, Tuple, Iterable, Optional
from collections import OrderedDict
from dataclasses import dataclass

from torch import nn
from torch.nn import functional as F
from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# 随机种子 & 设备
# -------------------------------
def seed_everything(seed: int = 42) -> None:
    """设定随机种子以确保可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 如需更高性能可把 deterministic=False / benchmark=True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------
# 损失函数
# -------------------------------
class FocalLoss(nn.Module):
    """
    二分类/多分类 Focal Loss（通用版），默认与 CrossEntropy 接口对齐：
    inputs: [N, C] logits；targets: [N] int64 类别索引
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)                  # pt \in (0,1)
        focal = (1 - pt) ** self.gamma * ce_loss  # 核心项
        if self.alpha is not None:
            focal = self.alpha * focal
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


# -------------------------------
# 模型权重向量化 / 聚合
# -------------------------------
def flatten_state_dict(state: Dict[str, torch.Tensor]) -> np.ndarray:
    """把 state_dict 的所有参数按 key 排序后 flatten 成 1D numpy 向量。"""
    parts: List[np.ndarray] = []
    for k in sorted(state.keys()):
        v = state[k].detach().cpu().numpy().ravel()
        parts.append(v)
    return np.concatenate(parts, axis=0) if parts else np.array([], dtype=np.float32)


def state_dict_to_cpu(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """拷贝一个 state_dict 并移动到 CPU（便于聚合/保存）。"""
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def fedavg(states: List[Dict[str, torch.Tensor]], sizes: Iterable[int]) -> OrderedDict:
    """
    经典 FedAvg：按客户端样本数加权求和。
    states:  每个客户端的 state_dict（需在同一结构）
    sizes:   每个客户端对应的样本权重（通常是节点/样本数）
    """
    sizes = list(sizes)
    total = float(sum(sizes))
    assert len(states) == len(sizes) and len(states) > 0, "FedAvg 参数数量不匹配"

    agg = OrderedDict()
    for k in states[0].keys():
        # 按权重求和
        s = None
        for i, st in enumerate(states):
            w = st[k].detach().cpu()
            coef = sizes[i] / total
            s = w * coef if s is None else s + w * coef
        agg[k] = s
    return agg


# -------------------------------
# 图相关工具
# -------------------------------
def knn_edges(sim: np.ndarray, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    根据相似度矩阵构造有向 kNN 边。
    返回:
        edge_index: [2, E] (send, recv)
        edge_weight: [E]
    """
    assert sim.ndim == 2 and sim.shape[0] == sim.shape[1], "sim 必须是方阵"
    idx = np.argsort(-sim, axis=1)[:, 1:k + 1]  # 每行取前 k 个非自身邻居
    send, recv = [], []
    for i, row in enumerate(idx):
        for j in row:
            send.append(i)
            recv.append(j)
    edge_index = torch.tensor([send, recv], dtype=torch.long)
    edge_weight = torch.tensor(sim[send, recv], dtype=torch.float32)
    return edge_index, edge_weight


# -------------------------------
# Δw 相似度 & 自动分簇（谱聚类）
# -------------------------------
@dataclass
class ClusterResult:
    labels: np.ndarray            # [num_clients] 聚类标签
    best_k: int                   # 自动选择的簇数
    silhouette: float             # 最佳轮廓系数
    sim: np.ndarray               # 客户端相似度矩阵（基于 Δw 的余弦相似度）
    dist: np.ndarray              # 距离矩阵 1 - sim
    delta_matrix: np.ndarray      # [num_clients, D]：每个客户端 Δw 向量


def cosine_sim_from_deltas(
    global_state: Dict[str, torch.Tensor],
    client_states: List[Dict[str, torch.Tensor]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    基于 Δw = w_client - w_global 计算客户端间的余弦相似度（用于聚类）。
    返回 (sim, dist, delta_matrix)
    """
    g_vec = flatten_state_dict(global_state)
    deltas = []
    for st in client_states:
        c_vec = flatten_state_dict(st)
        deltas.append(c_vec - g_vec)
    delta_matrix = np.vstack(deltas)  # [K, D]

    sim = cosine_similarity(delta_matrix)        # [K, K]
    sim = np.clip(sim, 0, None)                  # 只保留非负相似度
    # 归一化到 [0,1]
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)

    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return sim, dist, delta_matrix


def auto_cluster_clients(
    sim: np.ndarray,
    dist: Optional[np.ndarray] = None,
    max_k: int = 20,
    random_state: int = 42,
) -> ClusterResult:
    """
    在 [2, max_k] 自动搜索最佳簇数（按 silhouette_score 最大化），
    使用谱聚类（相似度矩阵作为亲和度）。
    """
    assert sim.ndim == 2 and sim.shape[0] == sim.shape[1], "sim 必须是方阵"
    num_clients = sim.shape[0]
    max_k = max(2, min(max_k, num_clients - 1))
    if dist is None:
        dist = 1.0 - sim
        np.fill_diagonal(dist, 0.0)

    best_k, best_score, best_labels = 2, -1.0, None
    for k in range(2, max_k + 1):
        labels_k = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=random_state
        ).fit_predict(sim)

        score = silhouette_score(dist, labels_k, metric="precomputed")
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels_k

    return ClusterResult(
        labels=best_labels,
        best_k=best_k,
        silhouette=float(best_score),
        sim=sim,
        dist=dist,
        delta_matrix=None  # 上层如需可再填充（见 cluster_from_states）
    )


def cluster_from_states(
    global_state: Dict[str, torch.Tensor],
    client_states: List[Dict[str, torch.Tensor]],
    max_k: int = 20,
    random_state: int = 42,
) -> ClusterResult:
    """
    便捷封装：直接从 {全局, 客户端} state_dict 计算 Δw → sim → 自动分簇。
    """
    sim, dist, delta_mat = cosine_sim_from_deltas(global_state, client_states)
    result = auto_cluster_clients(sim, dist, max_k=max_k, random_state=random_state)
    result.delta_matrix = delta_mat
    return result


# -------------------------------
# 统计/评估辅助
# -------------------------------
def weighted_average(values: List[float], weights: List[int]) -> float:
    """按权重求加权平均。"""
    if not values:
        return 0.0
    w = float(sum(weights))
    return sum(v * s for v, s in zip(values, weights)) / (w + 1e-12)


def numpy_nanmean(arr: List[float]) -> float:
    """安全的 nanmean。"""
    return float(np.nanmean(np.array(arr, dtype=float))) if len(arr) > 0 else float("nan")
