# server.py
from __future__ import annotations

import copy
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


class Server:
    """
    联邦学习“服务器”：
    - 维护一个全局模型权重
    - 执行聚合（FedAvg / 基于 Δw 的簇内聚合）
    - 记录并恢复最佳全局权重
    """

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device
        self.best_val: float = -1e18
        self.best_state: Dict[str, torch.Tensor] = copy.deepcopy(model.state_dict())

    # -----------------------------
    # 基础工具
    # -----------------------------
    @staticmethod
    def flatten_state_dict(state: Dict[str, torch.Tensor]) -> np.ndarray:
        """把 state_dict 展平为 1D 向量（按 key 排序，便于余弦相似度计算）"""
        parts: List[np.ndarray] = []
        for k in sorted(state.keys()):
            parts.append(state[k].detach().cpu().numpy().ravel())
        return np.concatenate(parts, axis=0)

    @staticmethod
    def fedavg(states: List[Dict[str, torch.Tensor]], sizes: List[int]) -> Dict[str, torch.Tensor]:
        """标准 FedAvg：按样本数加权求和"""
        assert len(states) == len(sizes) and len(states) > 0
        total = float(sum(sizes))
        agg = copy.deepcopy(states[0])
        for k in agg.keys():
            agg[k] = sum(states[i][k] * (sizes[i] / total) for i in range(len(states)))
        return agg

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return copy.deepcopy(self.model.state_dict())

    # -----------------------------
    # 聚合策略
    # -----------------------------
    def aggregate_fedavg(
        self,
        client_states: List[Dict[str, torch.Tensor]],
        sizes: List[int],
    ) -> Dict[str, torch.Tensor]:
        """
        直接 FedAvg 聚合
        """
        new_global = self.fedavg(client_states, sizes)
        self.model.load_state_dict(new_global)
        return new_global

    def aggregate_clustered(
        self,
        global_state: Dict[str, torch.Tensor],
        client_states: List[Dict[str, torch.Tensor]],
        sizes: List[int],
        *,
        max_k: int = 20,
        random_state: int = 42,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
        """
        基于“客户端权重增量 Δw = w_i - w_global”的谱聚类聚合：
          1) 计算每个客户端 Δw，基于余弦相似度得到 (N×N) 相似度矩阵
          2) 自动搜索最优簇数 k（2..max_k），以轮廓系数为准
          3) 簇内做 FedAvg，最后对所有簇模型再做一次加权平均（得到新的全局）
        返回：
          new_global_state, info(dict) 其中包含 labels / k / silhouette / cluster_models
        """
        n_clients = len(client_states)
        assert n_clients >= 2, "至少需要两个客户端进行簇聚合"

        # ---- 1) 计算 Δw 并构造相似度 / 距离矩阵 ----
        g_vec = self.flatten_state_dict(global_state)
        deltas = []
        for st in client_states:
            v = self.flatten_state_dict(st)
            deltas.append(v - g_vec)
        delta_mat = np.vstack(deltas)  # [N, D]

        sim = cosine_similarity(delta_mat)               # [N, N], [-1,1]
        sim = np.clip(sim, 0, None)                      # 负相似置0
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)  # 归一化到[0,1]
        dist = 1.0 - sim
        np.fill_diagonal(dist, 0.0)

        # ---- 2) 自动选择最佳簇数 k ----
        best_k, best_score = 2, -1.0
        max_k = min(max_k, n_clients - 1)
        for k in range(2, max_k + 1):
            labels_k = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=random_state,
            ).fit_predict(sim)

            score = silhouette_score(dist, labels_k, metric="precomputed")
            if score > best_score:
                best_k, best_score = k, score

        # ---- 3) 用最佳 k 重新分簇并做簇内聚合 ----
        clustering = SpectralClustering(
            n_clusters=best_k,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=random_state,
        )
        labels = clustering.fit_predict(sim)

        cluster_states: Dict[int, List[Dict[str, torch.Tensor]]] = defaultdict(list)
        cluster_sizes: Dict[int, List[int]] = defaultdict(list)
        for i, st in enumerate(client_states):
            cid = int(labels[i])
            cluster_states[cid].append(st)
            cluster_sizes[cid].append(sizes[i])

        cluster_models: Dict[int, Dict[str, torch.Tensor]] = {
            cid: self.fedavg(cluster_states[cid], cluster_sizes[cid]) for cid in cluster_states
        }

        # ---- 4) 将各簇模型再做一次加权平均 → 新的全局 ----
        new_global = self.fedavg(
            list(cluster_models.values()),
            [sum(cluster_sizes[cid]) for cid in cluster_models.keys()],
        )
        self.model.load_state_dict(new_global)

        info = {
            "labels": labels,                    # 每个客户端的簇编号
            "k": best_k,                         # 选择的簇数
            "silhouette": float(best_score),     # 轮廓系数
            "cluster_models": cluster_models,    # 各簇的聚合权重
        }
        return new_global, info

    def step(
        self,
        client_states: List[Dict[str, torch.Tensor]],
        sizes: List[int],
        *,
        mode: str = "cluster",  # 'cluster' 或 'fedavg'
        global_state: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
        """
        单轮聚合入口：
          - mode='fedavg'  → 直接 FedAvg
          - mode='cluster' → 需要传入 global_state，执行簇内聚合
        返回：(new_global_state, info)
        """
        if mode == "fedavg":
            new_global = self.aggregate_fedavg(client_states, sizes)
            return new_global, {"mode": "fedavg"}
        elif mode == "cluster":
            if global_state is None:
                raise ValueError("cluster 模式需要提供 global_state 以计算 Δw。")
            return self.aggregate_clustered(global_state, client_states, sizes, **kwargs)
        else:
            raise ValueError(f"未知聚合模式：{mode}")

    # -----------------------------
    # 早停 / 最优权重
    # -----------------------------
    def update_best(self, val_metric: float) -> None:
        """若当前轮的验证指标更优，则缓存当前全局权重为 best"""
        if val_metric > self.best_val:
            self.best_val = val_metric
            self.best_state = copy.deepcopy(self.model.state_dict())

    def load_best_state(self) -> None:
        """恢复至最优全局权重"""
        self.model.load_state_dict(self.best_state)
