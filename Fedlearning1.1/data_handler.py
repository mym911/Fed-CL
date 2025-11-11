# data_handler.py
from pathlib import Path
from typing import Optional, List, Any, Dict

import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances

from utils import logger, device
from transformers import WindowFusionTransformerSimple


class DataHandler:
    """
    负责：
      1) 读取 CSV 时间序列并构造多窗口动态功能连接(DFC)
      2) 用简化版 Transformer 对窗口序列做融合，得到每个被试的向量表征
      3) 构建“被试图”(subject graph)：基于 AAL/Schaefer 两套特征 + 表型信息 的融合相似度
    """
    def __init__(self,
                 ts_dir: Path,
                 atlas_type: str,
                 k: int = 10,
                 sigma: float = 1.0,
                 device: torch.device = device,
                 pca_components: int = 120,
                 num_heads: int = 4):
        self.atlas_type = atlas_type
        self.k = k
        self.ts_dir = ts_dir
        self.sigma = sigma
        self.device = device
        self.num_heads = num_heads

        # 延迟初始化的模块
        self.token_projection: Optional[nn.Linear] = None
        self.window_fusion_transformer: Optional[WindowFusionTransformerSimple] = None

        # 可选：标准化与PCA
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.pca_components = pca_components
        self.scaler_fitted = False
        self.pca_fitted = False

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    # ----------------------------------------------------------------------
    # 可选：特征矩阵的 scaler / pca 拟合与保存、加载
    # ----------------------------------------------------------------------
    def fit_and_save_scaler_pca(self, features_matrix: np.ndarray, scaler_path: Path, pca_path: Path):
        self.scaler = StandardScaler().fit(features_matrix)
        feats_scaled = self.scaler.transform(features_matrix)
        self.pca = PCA(n_components=self.pca_components).fit(feats_scaled)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.pca, pca_path)
        self.scaler_fitted = True
        self.pca_fitted = True
        logger.info(f"Scaler/PCA 保存到 {scaler_path} 和 {pca_path}")

    def load_scaler_pca(self, scaler_path: Path, pca_path: Path):
        self.scaler = joblib.load(scaler_path)
        self.pca = joblib.load(pca_path)
        self.scaler_fitted = True
        self.pca_fitted = True
        logger.info(f"已加载 Scaler/PCA：{scaler_path} / {pca_path}")

    # ----------------------------------------------------------------------
    # 读取 ABIDE 表型 + 时间序列，构造每个被试的动态特征向量
    # 返回: features_dict_raw, phenotypes_dict, labels_dict, 成功样本ID列表
    # ----------------------------------------------------------------------
    def preprocess_abide_data(self,
                              phenotypic_csv_path: Path,
                              ts_dir: Path,
                              pca_components: int = 120,
                              num_subjects: Optional[int] = None,
                              atlas: str = 'aal',
                              selected_subjects: Optional[List[Any]] = None):
        try:
            ph = pd.read_csv(phenotypic_csv_path)
        except Exception as e:
            logger.error(f"加载表型 CSV 失败: {e}")
            return {}, {}, {}, []

        required = ['SUB_ID', 'DX_GROUP', 'AGE_AT_SCAN', 'SEX']
        ph.columns = ph.columns.str.strip()
        for c in required:
            if c not in ph.columns:
                logger.error(f"表型 CSV 缺失列: {c}")
                return {}, {}, {}, []

        if selected_subjects:
            subjects = selected_subjects[:num_subjects] if num_subjects else selected_subjects
        else:
            all_ids = ph['SUB_ID'].astype(str).tolist()
            subjects = all_ids[:num_subjects] if num_subjects else all_ids

        features_dict_raw, phenotypes_dict, labels_dict = {}, {}, []
        features_dict_raw, phenotypes_dict, labels_dict = {}, {}, {}

        # 两组窗口（可按需调整）
        w1, s1 = 30, 5
        w2, s2 = 40, 15

        def fcn_to_mat(window: np.ndarray) -> np.ndarray:
            with np.errstate(divide='ignore', invalid='ignore'):
                fc = np.corrcoef(window, rowvar=False)
                fc = np.clip(fc, -0.999999, 0.999999)
                fc = np.arctanh(fc)
                fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
            return fc

        kept_ids = []
        for sid in subjects:
            csv_path = ts_dir / f"{sid}.csv"
            if not csv_path.exists():
                continue

            ts = pd.read_csv(csv_path, header=0).values  # (T, n_roi)
            if ts.shape[0] < max(w1, w2):
                continue

            windows1 = [ts[i:i + w1] for i in range(0, ts.shape[0] - w1 + 1, s1)]
            windows2 = [ts[i:i + w2] for i in range(0, ts.shape[0] - w2 + 1, s2)]
            mats1 = [fcn_to_mat(w) for w in windows1]
            mats2 = [fcn_to_mat(w) for w in windows2]

            f1 = self.dynamic_feature_aggregation(mats1, method='seq_transformer').squeeze(0)
            f2 = self.dynamic_feature_aggregation(mats2, method='seq_transformer').squeeze(0)
            fused_vec = np.concatenate([f1, f2], axis=0)
            features_dict_raw[sid] = fused_vec

            row = ph[ph['SUB_ID'].astype(str) == sid]
            if row.empty:
                continue
            row = row.iloc[0]
            age, sex = row['AGE_AT_SCAN'], row['SEX']
            phenotypes_dict[sid] = [age, sex]

            # ABIDE: DX_GROUP==1: Control, 2: Patient
            raw_label = row['DX_GROUP']
            labels_dict[sid] = 0 if raw_label == 1 else 1
            kept_ids.append(sid)

        return features_dict_raw, phenotypes_dict, labels_dict, kept_ids

    # ----------------------------------------------------------------------
    # 动态窗口序列 -> 特征向量（默认seq_transformer：上三角→线性投影→Transformer→池化）
    # ----------------------------------------------------------------------
    def dynamic_feature_aggregation(self, windowed_fcns: list, method: str = 'seq_transformer') -> np.ndarray:
        if not windowed_fcns:
            return np.zeros((1, 1), dtype=np.float32)

        if method == 'seq_transformer':
            triu = np.triu_indices_from(windowed_fcns[0], k=1)
            seq = np.stack([fc[triu] for fc in windowed_fcns], axis=0)  # (T, D)
            T, D = seq.shape

            tokens = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,T,D)

            # 懒加载投影层：D -> 64
            if self.token_projection is None or self.token_projection.in_features != D:
                self.token_projection = nn.Linear(D, 64).to(self.device)

            z = self.token_projection(tokens)  # (1,T,64)

            # 懒加载 Transformer
            if self.window_fusion_transformer is None:
                self.window_fusion_transformer = WindowFusionTransformerSimple(
                    input_dim=64, fused_dim=512, num_layers=2, n_heads=4, dropout=0.1
                ).to(self.device)

            fused = self.window_fusion_transformer(z)  # (1,512)
            return fused.detach().cpu().numpy()

        elif method == 'mean':
            triu = np.triu_indices_from(windowed_fcns[0], k=1)
            feats = [fc[triu] for fc in windowed_fcns]
            return np.mean(feats, axis=0).reshape(1, -1)

        else:
            raise ValueError("method 必须是 'seq_transformer' 或 'mean'")

    # ----------------------------------------------------------------------
    # 相似度构建：被试特征 / 表型
    # ----------------------------------------------------------------------
    def construct_subject_similarity(self, features_matrix: np.ndarray) -> np.ndarray:
        """RBF(高斯核)的欧氏距离相似度，主对角置1。"""
        pairwise_dist = pairwise_distances(features_matrix, metric='euclidean')
        sigma = np.percentile(pairwise_dist[pairwise_dist > 0], 25)
        sim = np.exp(-pairwise_dist ** 2 / (2 * (sigma + 1e-9) ** 2))
        np.fill_diagonal(sim, 1.0)
        return sim

    def construct_phenotype_similarity(self, phenotypes_array: np.ndarray) -> np.ndarray:
        """年龄(归一化绝对差) + 性别(相同0/不同1) -> Gower距离 -> 相似度=1-距离。"""
        n = phenotypes_array.shape[0]
        if n == 0:
            return np.array([[]], dtype=np.float32)

        ages = phenotypes_array[:, 0].astype(float)
        sexes = phenotypes_array[:, 1]

        min_age, max_age = ages.min(), ages.max()
        age_range = max(max_age - min_age, 1.0)
        age_dist = np.abs(ages[:, None] - ages[None, :]) / age_range
        sex_dist = (sexes[:, None] != sexes[None, :]).astype(np.float32)

        gower = (age_dist + sex_dist) / 2.0
        return 1.0 - gower

    # ----------------------------------------------------------------------
    # 构建群体图（被试图）
    # ----------------------------------------------------------------------
    def build_group_graph(self,
                          feat_aal: Dict[str, np.ndarray],
                          feat_sch: Dict[str, np.ndarray],
                          phenos: Dict[str, List[Any]],
                          labels: Dict[str, int],
                          k: Optional[int] = None) -> Data:
        if k is None:
            k = self.k

        subject_ids = sorted(feat_aal.keys())
        N = len(subject_ids)
        assert N > 0, "没有被试样本用于构图。"

        mat_aal = np.vstack([feat_aal[sid] for sid in subject_ids])   # (N, d_aal)
        mat_sch = np.vstack([feat_sch[sid] for sid in subject_ids])   # (N, d_sch)
        mat_ph  = np.array([phenos[sid] for sid in subject_ids])      # (N, 2)

        # 三路相似度
        S_aal = self.construct_subject_similarity(mat_aal)
        S_sch = self.construct_subject_similarity(mat_sch)
        S_phi = self.construct_phenotype_similarity(mat_ph)

        fused = S_aal * S_sch * S_phi

        # KNN 边
        edge_index, edge_weight = self._build_knn_edges(fused, k)

        # 节点特征：拼接 AAL+SCH
        x = torch.tensor(np.hstack([mat_aal, mat_sch]), dtype=torch.float32, device=self.device)
        y = torch.tensor([labels[sid] for sid in subject_ids], dtype=torch.long, device=self.device)

        data = Data(x=x,
                    edge_index=edge_index.to(self.device),
                    edge_weight=edge_weight.to(self.device),
                    y=y)
        data.subject_ids = subject_ids
        return data

    # ----------------------------------------------------------------------
    # 从相似度矩阵取每行 top-k 近邻生成有向边 (i -> j)
    # ----------------------------------------------------------------------
    def _build_knn_edges(self, similarity_matrix: np.ndarray, k: Optional[int] = None):
        if k is None:
            k = min(30, int(np.sqrt(similarity_matrix.shape[0])))

        sim = (similarity_matrix - similarity_matrix.min()) \
              / (similarity_matrix.max() - similarity_matrix.min() + 1e-8)

        idx = np.argsort(-sim, axis=1)[:, 1:k + 1]  # 跳过自己(第0)
        send, recv = [], []
        for i, nbrs in enumerate(idx):
            for j in nbrs:
                send.append(i)
                recv.append(j)

        edge_index = torch.tensor([send, recv], dtype=torch.long)
        edge_weight = torch.tensor(sim[send, recv], dtype=torch.float32)
        return edge_index, edge_weight
