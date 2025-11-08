# data.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='nilearn')

from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from nilearn.input_data import NiftiLabelsMasker
from nilearn import datasets

from utils import logger, device

# 可选：AAL 自定义标签（如需映射/校验时使用）
AAL_CUSTOM_LABELS = [
    '2001','2002','2101','2102','2111','2112','2201','2202','2211','2212',
    '2301','2302','2311','2312','2321','2322','2331','2332','2401','2402',
    '2501','2502','2601','2602','2611','2612','2701','2702','3001','3002',
    '4001','4002','4011','4012','4021','4022','4101','4102','4111','4112',
    '4201','4202','5001','5002','5011','5012','5021','5022','5101','5102',
    '5201','5202','5301','5302','5401','5402','6001','6002','6101','6102',
    '6201','6202','6211','6212','6221','6222','6301','6302','6401','6402',
    '7001','7002','7011','7012','7021','7022','7101','7102','8101','8102',
    '8111','8112','8121','8122','8201','8202','8211','8212','8301','8302',
    '9001','9002','9011','9012','9021','9022','9031','9032','9041','9042',
    '9051','9052','9061','9062','9071','9072','9081','9082','9100','9110',
    '9120','9130','9140','9150','9160','9170'
]

def collate_fn_with_dynamic_padding(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将同一 batch 内不同长度的时间序列动态右侧填充到最大长度。
    输入: [(features[1,n_rois,T_i], mask[n_rois,T_i], cond[cond_dim]), ...]
    输出: features[B,n_rois,T_max], masks[B,n_rois,T_max], conds[B,cond_dim]
    """
    max_time = max(features.shape[-1] for features, _, _ in batch)
    B = len(batch)
    n_rois = batch[0][0].shape[1]
    cond_dim = batch[0][2].shape[0]

    padded_features = torch.zeros((B, n_rois, max_time))
    padded_masks = torch.zeros((B, n_rois, max_time), dtype=torch.bool)
    conds = torch.zeros((B, cond_dim))

    for i, (features, mask, cond) in enumerate(batch):
        T = features.shape[-1]
        padded_features[i, :, :T] = features.squeeze(0)
        padded_masks[i, :, :T] = mask.squeeze(0)
        conds[i] = cond

    return padded_features.to(device), padded_masks.to(device), conds.to(device)


class ABIDEDataset(Dataset):
    """把内存中的 {sid: data} 字典适配成 PyTorch Dataset。"""
    def __init__(self, data_dict: Dict, phenotypic_data: Dict[str, np.ndarray]):
        self.data_dict = data_dict
        self.phenotypic_data = phenotypic_data
        self._keys = list(self.data_dict.keys())

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, idx: int):
        subj_id = self._keys[idx]
        subj_data = self.data_dict[subj_id]
        features = subj_data['session_1']['rest_features']          # (n_rois, T)
        roi_available_mask = subj_data['session_1']['missing_mask'] # (n_rois,), 1=可用 0=缺失

        T = features.shape[-1]
        mask_2d = (np.tile(roi_available_mask[:, np.newaxis], (1, T)) == 1)  # (n_rois, T) True=可用
        mask = torch.tensor(mask_2d, dtype=torch.bool)
        cond = torch.tensor(self.phenotypic_data[subj_id], dtype=torch.float32)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1,n_rois,T)
        return features_tensor, mask, cond


class ABIDEDataLoader:
    """
    负责从 CSV 读取 (n_rois x T) 的时间序列矩阵，并对接表型数据。
    目录结构：train_dir/<ScanDir ID>.csv, test_dir/<ScanDir ID>.csv
    """
    def __init__(self, config, global_mean=None, global_std=None):
        self.config = config
        self.phenotypes = self._load_phenotypic_data()

        # 通过一个样例 CSV 推断 ROI 数
        example_csv = next(self.config.roots["train"].glob("*.csv"), None)
        if not example_csv:
            raise RuntimeError(f"{self.config.roots['train']} 下没有任何 CSV 文件！")
        try:
            n_rois = pd.read_csv(example_csv, header=0).shape[1]
        except Exception:
            n_rois = pd.read_csv(example_csv, header=None).shape[1]
        self.custom_labels = list(range(n_rois))

        # 可选的全局归一化参数
        self.global_mean = global_mean
        self.global_std = global_std

        # 预留：若需要从 NIfTI 直接提取，可使用 masker（当前读取 CSV，不强依赖）
        if self.config.atlas_type == "aal":
            self.atlas = datasets.fetch_atlas_aal(version='SPM12')
        elif self.config.atlas_type == "schaefer200":
            self.atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200, resolution_mm=2)
        else:
            raise ValueError("Unsupported atlas type: " + str(self.config.atlas_type))

        self.masker = NiftiLabelsMasker(
            labels_img=self.atlas.maps,
            standardize=True, detrend=True,
            low_pass=0.1, high_pass=0.01, t_r=2,
            memory='nilearn_cache', verbose=0,
            resampling_target='data'
        )

    def _load_phenotypic_data(self) -> Dict[str, np.ndarray]:
        """读取表型 CSV，返回 {sid: np.array([age, sex])}，sex: 男=1.0 女=0.0/缺失=0.0。"""
        df = pd.read_csv(self.config.phenotypic_csv)
        df.columns = [c.strip() for c in df.columns]

        required_columns = ['ScanDir ID', 'Age', 'Gender']
        missing = set(required_columns) - set(df.columns)
        if missing:
            logger.error(f"表型数据中缺少列：{missing}")
            return {}

        phenotypes: Dict[str, np.ndarray] = {}
        for _, row in df.iterrows():
            sid = str(row['ScanDir ID'])
            age = float(row['Age'])

            g = row['Gender']
            if pd.isna(g):
                sex = 0.0
            elif isinstance(g, str):
                sex = 1.0 if g.upper().startswith('M') else 0.0
            else:
                try:
                    sex = 1.0 if int(g) == 1 else 0.0  # 常见编码：1=男, 2=女
                except Exception:
                    sex = 0.0

            phenotypes[sid] = np.array([age, sex], dtype=np.float32)

        logger.info(f"Loaded phenotypes for {len(phenotypes)} subjects")
        return phenotypes

    def load_dataset(self, mode: str):
        """
        返回:
          dataset: {sid: {'session_1': {'rest_features': (n_rois,T), 'missing_mask': (n_rois,)}}}
          missing: List[sid] 未在对应目录找到 CSV 的受试者
        """
        root: Path = self.config.roots.get(mode)
        if root is None:
            raise ValueError(f"Unknown mode {mode!r}")

        dataset: Dict[str, Dict] = {}
        missing: List[str] = []

        for sid in self.phenotypes:
            data = self._load_subject(root, sid)
            if data:
                dataset[sid] = data
            else:
                missing.append(sid)

        return dataset, missing

    def _load_subject(self, root_dir: Path, subject_id: str):
        """
        从 root_dir/{subject_id}.csv 读取 (T × n_rois) 或 (n_rois × T)（此处假设列为 ROI；按需调整）
        当前实现：假设 CSV 行为时间步，列为 ROI，因此转置为 (n_rois, T)。
        缺失 ROI：整列为 0 视为缺失（missing_mask=0）
        """
        csv_path = root_dir / f"{subject_id}.csv"
        if not csv_path.exists():
            logger.warning(f"文件不存在: {csv_path}")
            return {}

        # 若你的 CSV 没有表头，可将 header=None
        df = pd.read_csv(csv_path, header=0)
        features = df.values.astype(np.float32).T  # (n_rois, T)

        # 哪些 ROI 全 0：视作缺失
        missing_mask = (~np.all(features == 0, axis=1)).astype(np.int32)  # 1=可用, 0=缺失

        return {
            'session_1': {
                'rest_features': features,
                'missing_mask':  missing_mask
            }
        }
