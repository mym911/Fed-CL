# data_handler.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances

from transformers import WindowFusionTransformerSimple
from config import PCA_COMPONENTS

__all__ = ["DataHandler"]

logger = logging.getLogger(__name__)
device_default = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataHandler:
    """
    负责：
      1) 读取表型与时序 CSV；
      2) 滑窗 → 相关矩阵 → 特征聚合(Transformer/均值等)；
      3) 计算被试相似度/表型相似度；
      4) 融合多图谱，构建 PyG Data 图；
      5) （可选）Scaler/PCA 的拟合与加载。
    """

    def __init__(
        self,
        ts_dir: Path,
        atlas_type: str,
        k: int = 10,
        sigma: float = 1.0,
        similarity_threshold: float = 0.0,
        device: torch.device = device_default,
        pca_components: int = PCA_COMPONENTS,
        num_heads: int = 4,
    ) -> None:
        self.ts_dir = Path(ts_dir)
        self.atlas_type = atlas_type
        self.k = int(k)
        self.sigma = float(sigma)
        self.similarity_threshold = float(similarity_threshold)
        self.device = device
        self.num_heads = int(num_heads)

        # 缓存/延迟初始化模块
        self.token_projection: Optional[nn.Linear] = None
        self.window_fusion_transformer: Optional[WindowFusionTransformerSimple] = None

        # 归一化 & 降维
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.pca_components = int(pca_components)
        self.scaler_fitted = False
        self.pca_fitted = False
        self.actual_pca_components: Optional[int] = None

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # -------------------- Scaler / PCA --------------------
    def fit_and_save_scaler_pca(
        self,
        features_matrix: np.ndarray,
        scaler_path: Path,
        pca_path: Path,
    ) -> None:
        """
        在训练集上一次性 fit Scaler 与 PCA，并保存到磁盘。
        """
        assert features_matrix.ndim == 2, "features_matrix 应为二维 [N, D]"
        scaler_path = Path(scaler_path)
        pca_path = Path(pca_path)

        self.scaler = StandardScaler().fit(features_matrix)
        feats_scaled = self.scaler.transform(features_matrix)
        self.pca = PCA(n_components=self.pca_components).fit(feats_scaled)

        self.actual_pca_components = int(getattr(self.pca, "n_components_", self.pca_components))
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.pca, pca_path)

        self.scaler_fitted = True
        self.pca_fitted = True
        logger.info(f"Scaler/PCA 已保存：{scaler_path} / {pca_path}（PCA维度={self.actual_pca_components}）")

    def load_scaler_pca(self, scaler_path: Path, pca_path: Path) -> None:
        """
        推理阶段：加载已拟合好的 Scaler/PCA。
        """
        scaler_path = Path(scaler_path)
        pca_path = Path(pca_path)
        self.scaler = joblib.load(scaler_path)
        self.pca = joblib.load(pca_path)
        self.scaler_fitted = True
        self.pca_fitted = True
        self.actual_pca_components = int(getattr(self.pca, "n_components_", self.pca_components))
        logger.info(f"已加载 Scaler/PCA：{scaler_path} / {pca_path}")

    # -------------------- 数据预处理 --------------------
    def preprocess_abide_data(
        self,
        phenotypic_csv_path: Path,
        ts_dir: Path,
        pca_components: int = PCA_COMPONENTS,
        num_subjects: Optional[int] = None,
        atlas: str = "aal",
        selected_subjects: Optional[List[Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, List[Any]], Dict[str, int], List[str]]:
        """
        读取表型 CSV 与每个受试者的时序 CSV：
          - 两套滑窗 (w=30/40, s=5/15)
          - 每窗构建 FC (Fisher-Z)
          - 用 Transformer 进行窗口级融合，得到固定维度向量
        返回：
          features_dict_raw: {subj: (D,) 向量}
          phenotypes_dict : {subj: [age, sex]}
          labels_dict     : {subj: 0/1}
          kept_subjects   : List[str]
        """
        ts_dir = Path(ts_dir)
        phenotypic_csv_path = Path(phenotypic_csv_path)

        try:
            phenotypic = pd.read_csv(phenotypic_csv_path)
        except Exception as e:
            logger.error(f"读取表型 CSV 失败：{e}")
            return {}, {}, {}, []

        phenotypic.columns = phenotypic.columns.str.strip()
        required_cols = ["SUB_ID", "DX_GROUP", "AGE_AT_SCAN", "SEX"]
        for c in required_cols:
            if c not in phenotypic.columns:
                logger.error(f"表型 CSV 缺少列：{c}")
                return {}, {}, {}, []

        # 决定要处理的受试者列表
        if selected_subjects is not None:
            subjects = selected_subjects[: num_subjects] if num_subjects else list(selected_subjects)
        else:
            all_ids = phenotypic["SUB_ID"].astype(str).tolist()
            subjects = all_ids[: num_subjects] if num_subjects else all_ids

        features_dict_raw: Dict[str, np.ndarray] = {}
        phenotypes_dict: Dict[str, List[Any]] = {}
        labels_dict: Dict[str, int] = {}
        kept: List[str] = []

        # 两组滑窗配置
        w1, s1 = 30, 5
        w2, s2 = 40, 15

        def fcn_to_mat(window: np.ndarray) -> np.ndarray:
            with np.errstate(divide="ignore", invalid="ignore"):
                fc = np.corrcoef(window, rowvar=False)
                fc = np.clip(fc, -0.999999, 0.999999)  # 避免 arctanh 溢出
                fc = np.arctanh(fc)
                fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
            return fc

        for sid in subjects:
            csv_path = ts_dir / f"{sid}.csv"
            if not csv_path.exists():
                logger.warning(f"缺少时序文件：{csv_path}，跳过")
                continue

            ts = pd.read_csv(csv_path, header=0).values  # [T, n_roi]
            if ts.shape[0] < max(w1, w2):
                logger.warning(f"{sid} 的时间序列过短（{ts.shape[0]} < {max(w1, w2)}），跳过")
                continue

            # 生成两组窗口
            wins1 = [ts[i : i + w1] for i in range(0, ts.shape[0] - w1 + 1, s1)]
            wins2 = [ts[i : i + w2] for i in range(0, ts.shape[0] - w2 + 1, s2)]

            mats1 = [fcn_to_mat(w) for w in wins1]
            mats2 = [fcn_to_mat(w) for w in wins2]

            # 窗口级 Transformer 融合（与你原始代码一致的默认方案）
            f1 = self.dynamic_feature_aggregation(mats1, method="seq_transformer").squeeze(0)
            f2 = self.dynamic_feature_aggregation(mats2, method="seq_transformer").squeeze(0)
            fused_vec = np.concatenate([f1, f2], axis=0)  # (D1 + D2,)

            # 表型与标签
            row = phenotypic.loc[phenotypic["SUB_ID"].astype(str) == str(sid)]
            if row.empty:
                logger.warning(f"{sid} 的表型不存在，跳过")
                continue
            row = row.iloc[0]
            age = float(row["AGE_AT_SCAN"])
            sex = row["SEX"]
            phenotypes_dict[str(sid)] = [age, sex]

            dx = int(row["DX_GROUP"])
            if dx == 1:
                label = 0
            elif dx == 2:
                label = 1
            else:
                logger.warning(f"{sid} 的 DX_GROUP 异常：{dx}，跳过")
                continue

            features_dict_raw[str(sid)] = fused_vec.astype(np.float32)
            labels_dict[str(sid)] = label
            kept.append(str(sid))

        logger.info(f"成功处理 {len(kept)} 名受试者")
        return features_dict_raw, phenotypes_dict, labels_dict, kept

    # -------------------- 窗口特征聚合 --------------------
    def dynamic_feature_aggregation(self, windowed_fcns: List[np.ndarray], method: str = "seq_transformer") -> np.ndarray:
        """
        将一组窗口的 FC 矩阵聚合成固定维度向量。
        支持：
          - 'upper'                 : 先均值，再取上三角向量
          - 'mean'                  : 直接对所有窗口的上三角向量求均值
          - 'mean_then_transformer' : 上三角序列 → 线性投影 → 简版Transformer → 融合
          - 'transformer'           : 同上，细节略不同
          - 'transformer_ts'        : 先在时间维上均值得到 (T, n_roi)，再 Transformer
          - 'seq_transformer'       : 默认；上三角序列 → 线性投影64维 → TransformerSimple
        返回 shape: (1, D_out)
        """
        if not windowed_fcns:
            return np.zeros((1, 1), dtype=np.float32)

        method = method.lower()
        if method == "upper":
            fc_mean = np.mean(windowed_fcns, axis=0)
            iu = np.triu_indices_from(fc_mean, k=1)
            v = fc_mean[iu]
            return v[None, :].astype(np.float32)

        if method == "mean":
            triu_feats = []
            for fc in windowed_fcns:
                iu = np.triu_indices_from(fc, k=1)
                triu_feats.append(fc[iu].ravel())
            v = np.mean(triu_feats, axis=0)
            return v[None, :].astype(np.float32)

        if method in {"mean_then_transformer", "transformer", "seq_transformer"}:
            iu = np.triu_indices_from(windowed_fcns[0], k=1)
            seq = np.stack([fc[iu] for fc in windowed_fcns], axis=0)  # [T, D]
            tokens = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, T, D]

            # 线性投影到 64 维（若 D 变化则重建）
            D = tokens.size(-1)
            if self.token_projection is None or self.token_projection.in_features != D:
                self.token_projection = nn.Linear(D, 64).to(self.device)
            z = self.token_projection(tokens)  # [1, T, 64]

            # 延迟初始化 Transformer
            if self.window_fusion_transformer is None:
                self.window_fusion_transformer = WindowFusionTransformerSimple(
                    input_dim=64, fused_dim=512, num_layers=2, n_heads=4, dropout=0.1
                ).to(self.device)

            fused = self.window_fusion_transformer(z)  # [1, 512]
            return fused.detach().cpu().numpy().astype(np.float32)

        if method == "transformer_ts":
            # (T, win_len, n_roi) → 在时间维(w)上均值 → (T, n_roi) → Transformer
            ts_stack = np.array(windowed_fcns)  # [T, win_len, n_roi]
            ts_mean = ts_stack.mean(axis=1)     # [T, n_roi]
            tokens = torch.tensor(ts_mean, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, T, n_roi]

            D = tokens.size(-1)
            # 用输入维 = n_roi 的 Transformer
            model = WindowFusionTransformerSimple(input_dim=D, fused_dim=128, num_layers=1, n_heads=2, dropout=0.1).to(
                self.device
            )
            fused = model(tokens)  # [1, 128]
            return fused.detach().cpu().numpy().astype(np.float32)

        raise ValueError(f"不支持的聚合方法：{method}")

    # -------------------- 相似度计算 --------------------
    def construct_subject_similarity(self, features_matrix: np.ndarray) -> np.ndarray:
        """
        RBF(Gaussian) 相似度：基于欧氏距离，σ 取非零距离的 25 分位数。
        """
        assert features_matrix.ndim == 2, f"特征矩阵应为二维 [N, D]，得到 {features_matrix.shape}"
        dist = pairwise_distances(features_matrix, metric="euclidean")
        sigma = np.percentile(dist[dist > 0], 25) if np.any(dist > 0) else 1.0
        sim = np.exp(-(dist ** 2) / (2 * (sigma + 1e-9) ** 2))
        np.fill_diagonal(sim, 1.0)
        return sim.astype(np.float32)

    def construct_phenotype_similarity(self, phenotypes_array: np.ndarray) -> np.ndarray:
        """
        简化表型相似度：年龄(归一化差异) + 性别(相同=0，不同=1) 的平均距离 → 相似度。
        """
        n = phenotypes_array.shape[0]
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32)

        ages = phenotypes_array[:, 0].astype(float)
        sexes = phenotypes_array[:, 1]

        min_age, max_age = ages.min(), ages.max()
        age_range = (max_age - min_age) if max_age > min_age else 1.0

        age_dist = np.zeros((n, n), dtype=np.float32)
        sex_dist = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                age_d = abs(ages[i] - ages[j]) / age_range
                s_d = 0.0 if sexes[i] == sexes[j] else 1.0
                age_dist[i, j] = age_dist[j, i] = age_d
                sex_dist[i, j] = sex_dist[j, i] = s_d

        gower = (age_dist + sex_dist) / 2.0
        S_phi = 1.0 - gower  # 距离 → 相似度
        np.fill_diagonal(S_phi, 1.0)
        return S_phi.astype(np.float32)

    # -------------------- 图构建 --------------------
    def build_group_graph(
        self,
        feat_aal: Dict[str, np.ndarray],
        feat_sch: Dict[str, np.ndarray],
        phenos: Dict[str, List[Any]],
        labels: Dict[str, int],
        k: Optional[int] = None,
    ) -> Data:
        """
        将 AAL / Schaefer 特征与表型融合，构建 KNN 图并返回 PyG Data：
          - 节点特征 x 为 [feat_aal, feat_sch] 拼接；
          - 边基于 fused_sim = S_aal * S_sch * S_phi 的 kNN；
          - y 为节点标签（0/1）。
        """
        if k is None:
            k = self.k

        subject_ids = sorted(feat_aal.keys())
        assert set(subject_ids) == set(feat_sch.keys()) == set(phenos.keys()) == set(
            labels.keys()
        ), "AAL/SCH/表型/标签 的子集不一致"

        # 堆叠矩阵
        X_aal = np.vstack([feat_aal[s] for s in subject_ids])
        X_sch = np.vstack([feat_sch[s] for s in subject_ids])
        P = np.array([phenos[s] for s in subject_ids])

        # 三种相似度
        S_aal = self.construct_subject_similarity(X_aal)
        S_sch = self.construct_subject_similarity(X_sch)
        S_phi = self.construct_phenotype_similarity(P)

        fused_sim = (S_aal * S_sch * S_phi).astype(np.float32)

        edge_index, edge_weight = self._build_knn_edges(fused_sim, k)

        x = torch.tensor(np.hstack([X_aal, X_sch]), dtype=torch.float32, device=self.device)
        y = torch.tensor([labels[s] for s in subject_ids], dtype=torch.long, device=self.device)

        data = Data(
            x=x,
            edge_index=edge_index.to(self.device),
            edge_weight=edge_weight.to(self.device),
            y=y,
        )
        data.subject_ids = subject_ids
        return data

    @staticmethod
    def build_full_graph(
        feats: np.ndarray,
        sim_mat: np.ndarray,
        y: np.ndarray,
        train_idx_local: List[int],
        val_idx_local: List[int],
        k: int = 10,
    ) -> Data:
        """
        （可选）基于已有相似度矩阵直接构图。
        """
        edge_index, edge_weight = DataHandler._knn_edges_from_sim(sim_mat, k)
        N = feats.shape[0]
        data = Data(
            x=torch.tensor(feats, dtype=torch.float32),
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=torch.tensor(y, dtype=torch.long),
        )
        # 可在上层根据需要设置 mask
        data.train_mask = torch.zeros(N, dtype=torch.bool)
        data.val_mask = torch.zeros(N, dtype=torch.bool)
        data.train_mask[train_idx_local] = True
        data.val_mask[val_idx_local] = True
        return data

    @staticmethod
    def _knn_edges_from_sim(sim: np.ndarray, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sim = sim.copy().astype(np.float32)
        # 归一化以稳定取 Top-k
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
        idx = np.argsort(-sim, axis=1)[:, 1 : k + 1]  # 排除自己
        send, recv = [], []
        for i, row in enumerate(idx):
            for j in row:
                send.append(i)
                recv.append(j)
        edge_index = torch.tensor([send, recv], dtype=torch.long)
        edge_weight = torch.tensor(sim[send, recv], dtype=torch.float32)
        return edge_index, edge_weight

    def _build_knn_edges(self, similarity_matrix: np.ndarray, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        固定 k 近邻的有向边（每个点连向相似度最高的 k 个邻居）。
        """
        return self._knn_edges_from_sim(similarity_matrix, k)
