# utils.py
import logging, random, os
import numpy as np
import torch
from typing import Tuple, Optional

# 全局 device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 全局 logger（主程里可再 basicConfig）
logger = logging.getLogger("brain_completion")

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def compute_global_normalization_params(train_dataset: dict, num_rois: int) -> Tuple[np.ndarray, np.ndarray]:
    """把你原函数搬过来，按 ROI 维度聚合后计算 mean/std。"""
    roi_data = [[] for _ in range(num_rois)]
    for subj_data in train_dataset.values():
        features = subj_data['session_1']['rest_features']
        assert features.shape[0] == num_rois, f"ROI数量不匹配: 预期{num_rois}，实际{features.shape[0]}"
        for roi_idx in range(num_rois):
            roi_time_series = features[roi_idx, :].flatten()
            roi_data[roi_idx].extend(roi_time_series.tolist())
    global_mean = np.zeros(num_rois); global_std = np.zeros(num_rois)
    for roi_idx in range(num_rois):
        data = np.array(roi_data[roi_idx])
        global_mean[roi_idx] = data.mean()
        global_std[roi_idx] = data.std()
    return global_mean, global_std

class DynamicMaskGenerator:
    """与你现有的 DynamicMaskGenerator 一致。"""
    def __init__(self, mask_rate: Optional[float] = None):
        self.mask_rate = mask_rate if mask_rate is not None else 0.3

    def generate_mask(self, shape, mode: Optional[str] = None) -> torch.Tensor:
        if mode is None: mode = 'roi'
        if mode != 'roi':
            raise ValueError(f"无效的掩蔽模式: {mode}")
        import torch
        if len(shape) == 2:
            n_rois = shape[0]
            mask = torch.zeros(shape, dtype=torch.bool)
            roi_mask = torch.rand(n_rois) < self.mask_rate
            mask[roi_mask, :] = True
            return mask
        elif len(shape) == 3:
            B, n_rois, T = shape
            mask = torch.zeros(shape, dtype=torch.bool)
            for i in range(B):
                roi_mask = torch.rand(n_rois) < self.mask_rate
                mask[i, roi_mask, :] = True
            return mask
        elif len(shape) == 4:
            B, C, n_rois, T = shape
            mask = torch.zeros(shape, dtype=torch.bool)
            for i in range(B):
                roi_mask = torch.rand(n_rois) < self.mask_rate
                mask[i, :, roi_mask, :] = True
            return mask
        else:
            raise ValueError("Unsupported tensor shape for ROI mask generation.")
