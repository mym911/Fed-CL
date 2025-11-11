# server.py
from __future__ import annotations
from typing import Dict, List, Tuple
import copy
import numpy as np
import torch

from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

from utils import logger, device, fedavg, flatten_state_dict


class Server:
    """
    联邦学习服务端：
      - 维护全局模型
      - FedAvg 参数聚合
      - 基于客户端更新的“相似度分簇 + 两段式聚合”（簇内 FedAvg → 簇间再聚合）
      - 记录与恢复最佳全局权重
    """
    def __init__(self, model: torch.nn.Module, device_: torch.device = device):
        self.model = model.to(device_)
        self.device = device_
        self.best_val = -np.inf
        self.best_state = copy.deepcopy(self.model.state_dict())

    # -------------------- 基础聚合 --------------------
    @staticmethod
    def fedavg(states: List[Dict[str, torch.Tensor]], sizes: List[int]) -> Dict[str, torch.Tensor]:
        """包装 utils.fedavg，返回聚合后的 state_dict。"""
        return fedavg(states, sizes)

    # -------------------- 分簇 + 两段式聚合 --------------------
    def cluster_aggregate(
        self,
        global_state: Dict[str, torch.Tensor],
        client_states: List[Dict[str, torch.Tensor]],
        sizes: List[int],
        num_clients: int,
        max_k: int = 20
    ) -> Tuple[Dict[int, Dict[str, torch.Tensor]], np.ndarray, int, float]:
        """
        1) 计算每个客户端相对全局的 Δw
        2) 以 Δw 的余弦相似度做谱聚类；自动选择 k（基于 silhouette）
        3) 簇内做 FedAvg，再把簇模型做一次加权平均，得到新的全局模型
        Returns:
            cluster_models: {cluster_id -> state_dict}
            labels:         每个客户端所属簇标签 (shape [num_clients])
            best_k:         选择的簇数
            best_score:     对应的 silhouette 得分
        """
        # --------- 边界处理 ---------
        if num_clients <= 1 or len(client_states) <= 1:
            logger.warning("客户端数量不足以分簇，直接 FedAvg。")
            agg = self.fedavg(client_states, sizes)
            self.model.load_state_dict(agg)
            return {0: agg}, np.zeros(len(client_states), dtype=int), 1, -1.0

        # --------- 1) 计算 Δw 相似度矩阵 ---------
        global_vec = flatten_state_dict(global_state)
        deltas = []
        for w in client_states:
            v = flatten_state_dict(w)
            deltas.append(v - global_vec)
        delta_mat = np.vstack(deltas)  # [N, D]

        sim = cosine_similarity(delta_mat)            # [-1, 1]
        sim = np.clip(sim, 0.0, None)                 # [0, 1]（负相似置0）
        if sim.max() > sim.min():
            sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)  # 归一化到 [0,1]
        np.fill_diagonal(sim, 1.0)
        dist = 1.0 - sim
        np.fill_diagonal(dist, 0.0)

        # --------- 2) 自动选择 k（silhouette on precomputed distance）---------
        best_k, best_score = 2, -1.0
        k_hi = max(2, min(max_k, num_clients))  # 至多不超过客户端数
        labels_best = None

        for k in range(2, k_hi + 1):
            try:
                sc = SpectralClustering(
                    n_clusters=k,
                    affinity='precomputed',
                    assign_labels='kmeans',
                    random_state=42
                )
                labels_k = sc.fit_predict(sim)
                score_k = silhouette_score(dist, labels_k, metric='precomputed')
                if score_k > best_score:
                    best_k, best_score = k, score_k
                    labels_best = labels_k
            except Exception as e:
                logger.warning(f"谱聚类 k={k} 失败：{e}")

        # 如果谱聚类全失败或结果退化，回退到单簇
        if labels_best is None:
            logger.warning("谱聚类失败，退化为单簇 FedAvg。")
            agg = self.fedavg(client_states, sizes)
            self.model.load_state_dict(agg)
            return {0: agg}, np.zeros(len(client_states), dtype=int), 1, -1.0

        # --------- 3) 簇内 FedAvg，簇间再聚合 ---------
        cluster_states = defaultdict(list)
        cluster_sizes = defaultdict(list)
        for idx, state in enumerate(client_states):
            cid = int(labels_best[idx])
            cluster_states[cid].append(state)
            cluster_sizes[cid].append(sizes[idx])

        cluster_models: Dict[int, Dict[str, torch.Tensor]] = {
            cid: self.fedavg(cluster_states[cid], cluster_sizes[cid])
            for cid in cluster_states.keys()
        }

        # 簇间聚合（按簇内样本和）
        new_global = self.fedavg(
            list(cluster_models.values()),
            [sum(cluster_sizes[cid]) for cid in cluster_models.keys()]
        )
        self.model.load_state_dict(new_global)

        return cluster_models, labels_best, int(best_k), float(best_score)

    # -------------------- 最优模型管理 --------------------
    def update_best(self, val_metric: float, state: Dict[str, torch.Tensor]):
        """若指标更优，则缓存最佳权重。"""
        if val_metric > self.best_val:
            self.best_val = float(val_metric)
            self.best_state = copy.deepcopy(state)

    def load_best_state(self):
        """加载历史最佳权重到 self.model。"""
        self.model.load_state_dict(self.best_state)
