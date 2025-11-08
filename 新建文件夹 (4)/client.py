# client.py
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

from config import (
    PHENOTYPIC_CSV_PATH,
    TS_DIR_AAL,
    TS_DIR_SCH,
    PCA_COMPONENTS,
    K,
    INIT_LR,
    WEIGHT_DECAY,
)
from data_handler import DataHandler
from models_gnn import SingleGraphModel
from utils import FocalLoss


logger = logging.getLogger(__name__)


class Client:
    """
    联邦学习中的“客户端”：
    - 本地预处理（加载 AAL / Sch 时序并提取特征、拼表型和标签）
    - 构建群体图（DataHandler.build_group_graph）
    - 本地训练（可选 FedProx 正则）
    - 本地评估
    """

    def __init__(
        self,
        client_id: int,
        raw_ids: List[str],
        test_ids_global: List[str],
        device: torch.device,
        *,
        k: int = K,
        lr: float = INIT_LR,
        batch_size: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        self.id = client_id
        self.raw_ids = list(raw_ids)
        self.test_ids_global = list(test_ids_global)
        self.device = device
        self.k = k
        self.lr = lr
        self.batch_size = batch_size

        # ① 本地预处理 → 四个字典（feat_aal, feat_sch, phenos, labels）
        self._run_local_preprocess()

        # ② 基于两套图谱 + 表型构建群体图
        dh = DataHandler(
            ts_dir=TS_DIR_AAL,
            atlas_type="aal",
            k=self.k,
            pca_components=PCA_COMPONENTS,
            device=self.device,
        )
        self.graph = dh.build_group_graph(
            feat_aal=self.feat_aal,
            feat_sch=self.feat_sch,
            phenos=self.phenos,
            labels=self.labels,
            k=self.k,
        )
        self.data = self.graph.x
        self.y_all = self.graph.y
        self.subject_ids = self.graph.subject_ids  # 排好序的 subject 列表
        self.sid2idx = {sid: i for i, sid in enumerate(self.subject_ids)}

        # ③ 训练 / 验证 / 测试 节点划分（以“节点索引”形式保存）
        self._make_local_split()

        # ④ DataLoader（只喂节点索引）
        self.loader = DataLoader(
            self.train_idx_local.cpu(), batch_size=self.batch_size, shuffle=True
        )

        # ⑤ 本地模型 + 损失
        in_dim = self.data.size(1)
        num_classes = len(set(self.labels.values()))
        self.model = SingleGraphModel(
            in_channels=in_dim, hidden_dim=hidden_dim, num_classes=num_classes
        ).to(self.device)

        # 可替换为 nn.CrossEntropyLoss()
        self.criterion: nn.Module = FocalLoss(gamma=2.0).to(self.device)

        logger.info(
            f"[Client{self.id}] init | nodes={self.graph.num_nodes} | "
            f"train={len(self.train_idx_local)} | val={len(self.val_idx_local)} | "
            f"test={len(self.test_idx_global)}"
        )

    # -----------------------------
    # 数据与图构建
    # -----------------------------
    def _run_local_preprocess(self) -> None:
        """
        读取 CSV → 生成四个字典，并确保 AAL / Sch 两套图谱都有样本。
        """
        dh_aal = DataHandler(
            TS_DIR_AAL, "aal", k=self.k, pca_components=PCA_COMPONENTS, device=self.device
        )
        dh_sch = DataHandler(
            TS_DIR_SCH, "schaefer200", k=self.k, pca_components=PCA_COMPONENTS, device=self.device
        )

        # 仅处理属于 {raw_ids ∪ test_ids_global} 的样本
        selected = self.raw_ids + self.test_ids_global

        self.feat_aal, self.phenos, self.labels, _ = dh_aal.preprocess_abide_data(
            PHENOTYPIC_CSV_PATH, TS_DIR_AAL, atlas="aal", selected_subjects=selected
        )
        feat_sch, _, _, _ = dh_sch.preprocess_abide_data(
            PHENOTYPIC_CSV_PATH, TS_DIR_SCH, atlas="schaefer200", selected_subjects=selected
        )
        self.feat_sch = feat_sch

        # 过滤：必须 AAL 和 Sch 都存在
        common = set(self.feat_aal) & set(self.feat_sch)
        dropped = set(selected) - common
        if dropped:
            logger.warning(
                f"[Client{self.id}] drop non-overlap subjects (AAL/Schaefer mismatch): {len(dropped)}"
            )

        self.feat_aal = {k: self.feat_aal[k] for k in common}
        self.feat_sch = {k: self.feat_sch[k] for k in common}
        self.phenos = {k: self.phenos[k] for k in common}
        self.labels = {k: self.labels[k] for k in common}

        if not self.labels:
            raise RuntimeError(f"[Client{self.id}] no usable subjects after filtering!")

        # 用过滤后的 id 覆盖 raw_ids，避免后续 split 中出现无效 id
        self.raw_ids = list(set(self.raw_ids) & common)

    def _make_local_split(self) -> None:
        """
        把“全体节点（subject_ids）”中属于本客户端训练域的样本划分为 train/val，
        并把“全局测试集 test_ids_global”映射为节点索引 test_idx_global。
        """
        # 排除全局测试 id 后，才是本地可训练的样本
        local_ids = [sid for sid in self.subject_ids if sid not in self.test_ids_global]

        y_local = [self.labels[sid] for sid in local_ids]
        tr_ids, val_ids = train_test_split(
            local_ids, test_size=0.2, random_state=42, stratify=y_local
        )

        # 将 subject id 映射到“整图上的节点索引”
        self.train_idx_local = torch.tensor(
            [self.sid2idx[sid] for sid in tr_ids], dtype=torch.long, device=self.device
        )
        self.val_idx_local = torch.tensor(
            [self.sid2idx[sid] for sid in val_ids], dtype=torch.long, device=self.device
        )
        self.test_idx_global = torch.tensor(
            [self.sid2idx[sid] for sid in self.test_ids_global if sid in self.sid2idx],
            dtype=torch.long,
            device=self.device,
        )

    # -----------------------------
    # 本地训练 / 评估
    # -----------------------------
    def local_train(
        self,
        epochs: int = 1,
        *,
        global_params: Optional[List[torch.Tensor]] = None,
        mu: float = 0.01,
    ) -> Tuple[Dict[str, torch.Tensor], int, float, float]:
        """
        进行本地训练。支持 FedProx 近端正则：
          loss += (mu/2) * Σ ||w_local - w_global||^2

        返回:
          state_dict_cpu, num_nodes, val_acc, val_loss
        """
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

        for _ in range(epochs):
            for batch_idx in self.loader:  # batch_idx 是一批“节点索引”
                batch_idx = batch_idx.to(self.device)

                _, logits = self.model(self.data, self.graph.edge_index, self.graph.edge_weight)
                loss = self.criterion(logits[batch_idx], self.y_all[batch_idx])

                # FedProx：加入近端正则
                if global_params is not None and mu > 0:
                    prox = 0.0
                    for p_local, p_global in zip(self.model.parameters(), global_params):
                        prox = prox + (p_local - p_global).pow(2).sum()
                    loss = loss + (mu / 2.0) * prox

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

        # 返回 CPU 版的 state_dict
        state_dict_cpu = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

        # 简单验证
        self.model.eval()
        with torch.no_grad():
            _, logits = self.model(self.data, self.graph.edge_index, self.graph.edge_weight)
            val_logits = logits[self.val_idx_local]
            val_y = self.y_all[self.val_idx_local]
            val_loss = self.criterion(val_logits, val_y).item()
            val_acc = (val_logits.argmax(dim=1) == val_y).float().mean().item()

        return state_dict_cpu, int(self.graph.num_nodes), float(val_acc), float(val_loss)

    def evaluate_local(self) -> Tuple[Dict[str, float], int]:
        """
        在全局测试索引上评估（只做一次，不参与训练）。
        返回: 指标字典和测试样本数量
        """
        if self.test_idx_global.numel() == 0:
            return {"acc": 0.0, "f1": 0.0, "auc": 0.0}, 0

        self.model.eval()
        with torch.no_grad():
            _, logits = self.model(self.graph.x, self.graph.edge_index, self.graph.edge_weight)
            probs = F.softmax(logits[self.test_idx_global], dim=1)
            preds = probs.argmax(dim=1)
            y = self.graph.y[self.test_idx_global]

        acc = (preds == y).float().mean().item()
        f1 = f1_score(y.cpu(), preds.cpu(), average="macro")
        try:
            auc = roc_auc_score(y.cpu(), probs[:, 1].cpu())
        except ValueError:
            auc = float("nan")

        return {"acc": acc, "f1": f1, "auc": float(auc)}, int(self.test_idx_global.numel())

    # -----------------------------
    # 与服务器交互的辅助方法
    # -----------------------------
    def load_global(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """从服务器下发的权重更新本地模型"""
        self.model.load_state_dict(state_dict)

    def update_graph(self, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> None:
        """
        （可选）若服务器构造了新的动态图，可在本地替换之。
        注意：x / y / subject_ids 不变。
        """
        self.graph.edge_index = edge_index.to(self.device)
        self.graph.edge_weight = edge_weight.to(self.device)
