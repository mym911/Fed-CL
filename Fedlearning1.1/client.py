# client.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional

import copy
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

from config import CFG
from utils import logger, device, FocalLoss
from data_handler import DataHandler
from models_gnn import SingleGraphModel


class Client:
    """
    单个联邦客户端：
      1) 本地预处理：从 AAL/Schaefer 两目录读取 {SUB_ID}.csv，做DFC + Transformer 融合，生成被试向量
      2) 构建群体图（被试图）：AAL/SCH/表型相似度融合 + KNN 边
      3) 本地训练：在本地图上训练 GNN（支持 FedProx 近端正则）
      4) 评估：在全局 test 集对应的节点上评估
    """
    def __init__(
        self,
        client_id: int,
        raw_ids: List[str],
        test_ids_global: List[str],
        device_: torch.device = device,
        k: int = CFG.K,
        lr: float = CFG.INIT_LR,
        batch_size: int = CFG.BATCH_SIZE,
        hidden_dim: int = 128,
        num_classes: int = 2,
        heads: int = 4,
        dropout: float = 0.5,
    ):
        self.id = client_id
        self.device = device_
        self.k = k
        self.lr = lr
        self.batch_size = batch_size

        # 需要在 split 时用到
        self._raw_ids_input = list(raw_ids)
        self.test_ids_global = list(test_ids_global)

        # 1) 本地预处理（构建 AAL/SCH/表型/标签）
        self._run_local_preprocess()

        # 2) 构图（被试图）
        dh = DataHandler(
            ts_dir=CFG.TS_DIR_AAL,
            atlas_type='aal',
            k=self.k,
            pca_components=CFG.PCA_COMPONENTS,
            device=self.device
        )
        self.graph = dh.build_group_graph(
            feat_aal=self.feat_aal,
            feat_sch=self.feat_sch,
            phenos=self.phenos,
            labels=self.labels,
            k=self.k
        )
        self.data = self.graph.x
        self.y_all = self.graph.y
        self.subject_ids = self.graph.subject_ids  # List[str]

        # 3) 训练/验证/测试划分（节点级索引）
        self._make_local_split()

        # 4) 小批索引 DataLoader（只装载训练节点索引）
        self.loader = DataLoader(
            list(self.train_idx_local.cpu().numpy()),
            batch_size=self.batch_size,
            shuffle=True
        )

        # 5) 本地模型与损失
        in_dim = self.data.size(1)
        self.model = SingleGraphModel(
            in_channels=in_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            heads=heads,
            dropout=dropout
        ).to(self.device)

        self.criterion = FocalLoss(gamma=2).to(self.device)
        # 如需普通CE：
        # self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    # --------------------------------------------------------------------- #
    # 预处理：从 AAL 与 SCH 两套时间序列生成被试表示；过滤只保留共有样本
    # --------------------------------------------------------------------- #
    def _run_local_preprocess(self):
        dh_aal = DataHandler(
            ts_dir=CFG.TS_DIR_AAL,
            atlas_type='aal',
            k=CFG.K,
            pca_components=CFG.PCA_COMPONENTS,
            device=self.device
        )
        dh_sch = DataHandler(
            ts_dir=CFG.TS_DIR_SCH,
            atlas_type='schaefer200',
            k=CFG.K,
            pca_components=CFG.PCA_COMPONENTS,
            device=self.device
        )

        selected = self._raw_ids_input + self.test_ids_global
        feat_aal, phenos, labels, _ = dh_aal.preprocess_abide_data(
            phenotypic_csv_path=CFG.PHENOTYPIC_CSV_PATH,
            ts_dir=CFG.TS_DIR_AAL,
            atlas='aal',
            selected_subjects=selected
        )
        feat_sch, _, _, _ = dh_sch.preprocess_abide_data(
            phenotypic_csv_path=CFG.PHENOTYPIC_CSV_PATH,
            ts_dir=CFG.TS_DIR_SCH,
            atlas='schaefer200',
            selected_subjects=selected
        )

        common = set(feat_aal) & set(feat_sch)
        if len(common) == 0:
            raise RuntimeError(f"[Client{self.id}] 本地无共同样本，无法构图。")

        # 过滤为共同被试
        self.feat_aal = {sid: feat_aal[sid] for sid in common}
        self.feat_sch = {sid: feat_sch[sid] for sid in common}
        self.phenos = {sid: phenos[sid] for sid in common}
        self.labels = {sid: labels[sid] for sid in common}

        # 更新 raw_ids 为有效的本地样本
        self._raw_ids_valid = [sid for sid in self._raw_ids_input if sid in common]

    # --------------------------------------------------------------------- #
    # 划分训练/验证/（全局）测试集 —— 都是“节点索引”
    # --------------------------------------------------------------------- #
    def _make_local_split(self):
        # 全体本地可训练样本（排除全局测试 ID）
        all_local_ids = [sid for sid in self.subject_ids if sid not in self.test_ids_global]
        y_local = [self.labels[sid] for sid in all_local_ids]

        if len(set(y_local)) < 2 or len(all_local_ids) < 3:
            # 样本或类别过少时退化为简单切分
            split = max(1, int(0.8 * len(all_local_ids)))
            tr_ids = all_local_ids[:split]
            val_ids = all_local_ids[split:]
        else:
            tr_ids, val_ids = train_test_split(
                all_local_ids,
                test_size=0.2,
                random_state=42,
                stratify=y_local
            )

        # 转为“在 subject_ids 中的节点索引”
        idx_map = {sid: i for i, sid in enumerate(self.subject_ids)}
        self.train_idx_local = torch.tensor([idx_map[s] for s in tr_ids], dtype=torch.long, device=self.device)
        self.val_idx_local = torch.tensor([idx_map[s] for s in val_ids], dtype=torch.long, device=self.device)
        self.test_idx_global = torch.tensor([idx_map[s] for s in self.subject_ids if s in self.test_ids_global],
                                            dtype=torch.long, device=self.device)

    # --------------------------------------------------------------------- #
    # 本地训练（可选 FedProx 近端项）
    # --------------------------------------------------------------------- #
    def local_train(self, epochs: int = 1,
                    global_params: Optional[List[torch.Tensor]] = None,
                    mu: float = 0.01) -> Tuple[Dict[str, torch.Tensor], int, float, float]:
        """
        Returns:
            state_dict_cpu: 本地训练后的模型参数（全cpu）
            num_nodes: 本地图节点数（用于加权）
            val_acc:   本地验证集准确率
            val_loss:  本地验证集loss
        """
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=CFG.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

        for _ in range(epochs):
            for batch_idx in self.loader:
                # DataLoader 返回的是 int 索引（CPU），转到 device
                if isinstance(batch_idx, list) or isinstance(batch_idx, tuple):
                    batch_idx = torch.tensor(batch_idx, dtype=torch.long, device=self.device)
                else:
                    batch_idx = torch.as_tensor(batch_idx, dtype=torch.long, device=self.device)

                _, logits = self.model(self.data, self.graph.edge_index, self.graph.edge_weight)
                loss = self.criterion(logits[batch_idx], self.y_all[batch_idx])

                # FedProx 近端正则
                if global_params is not None and mu > 0:
                    prox = 0.0
                    for p_local, p_global in zip(self.model.parameters(), global_params):
                        prox += (p_local - p_global).pow(2).sum()
                    loss = loss + (mu / 2.0) * prox

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

        # 验证
        with torch.no_grad():
            self.model.eval()
            _, logits = self.model(self.data, self.graph.edge_index, self.graph.edge_weight)
            val_logits = logits[self.val_idx_local]
            val_y = self.y_all[self.val_idx_local]
            val_loss = self.criterion(val_logits, val_y).item()
            val_acc = (val_logits.argmax(1) == val_y).float().mean().item()

        # 导出 CPU 权重
        state_dict_cpu = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        return state_dict_cpu, int(self.graph.num_nodes), float(val_acc), float(val_loss)

    # --------------------------------------------------------------------- #
    # 在全局测试节点上评估
    # --------------------------------------------------------------------- #
    def evaluate_local(self) -> Tuple[Dict[str, float], int]:
        if self.test_idx_global.numel() == 0:
            return {'acc': 0.0, 'f1': 0.0, 'auc': 0.0}, 0

        self.model.eval()
        with torch.no_grad():
            _, logits = self.model(self.graph.x, self.graph.edge_index, self.graph.edge_weight)
            probs = F.softmax(logits[self.test_idx_global], dim=1)
            preds = probs.argmax(1)
            y = self.graph.y[self.test_idx_global]

        acc = (preds == y).float().mean().item()
        f1 = f1_score(y.cpu(), preds.cpu(), average='macro')
        try:
            auc = roc_auc_score(y.cpu(), probs[:, 1].cpu())
        except ValueError:
            auc = float('nan')

        return {'acc': float(acc), 'f1': float(f1), 'auc': float(auc)}, int(self.test_idx_global.numel())

    # --------------------------------------------------------------------- #
    # 同步全局权重 / 动态更新图（如做第二阶段微调）
    # --------------------------------------------------------------------- #
    def load_global(self, state_dict: Dict[str, torch.Tensor]):
        self.model.load_state_dict(state_dict)

    def update_graph(self, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        self.graph.edge_index = edge_index.to(self.device)
        self.graph.edge_weight = edge_weight.to(self.device)
