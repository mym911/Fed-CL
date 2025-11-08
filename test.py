# test.py
import numpy as np
import torch
from pathlib import Path
from typing import Dict

from utils import device, logger
from data import ABIDEDataLoader
from models import ROICompletionGenerator


class BrainCompletionTester:
    def __init__(self, model_config, data_config):
        self.cfg = model_config
        self.data_cfg = data_config
        self.loader = ABIDEDataLoader(data_config)
        self._load_normalization_params()
        self.model = self._load_model()
        self.model.eval()

    def _load_normalization_params(self):
        """读取训练阶段保存的全局均值/方差，用于反标准化。"""
        norm_params_path = self.data_cfg.model_dir / "norm_params.npz"
        if norm_params_path.exists():
            npz = np.load(norm_params_path)
            self.loader.mean = npz["mean"]
            self.loader.std = npz["std"]
            logger.info("加载归一化参数成功")
        else:
            logger.warning("归一化参数文件不存在！将以未标准化方式输出。")

    def _load_model(self) -> ROICompletionGenerator:
        """加载最优生成器权重。"""
        num_rois = len(self.loader.custom_labels)
        model = ROICompletionGenerator(self.cfg, num_rois=num_rois).to(device)
        model_path = self.data_cfg.model_dir / "roi_completion_best.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    def test_with_real_missing(self) -> Dict[str, np.ndarray]:
        """
        对测试集进行补齐：
        - 把缺失 ROI 和“整列为 0 的 ROI”统一视为缺失并置零输入
        - 用模型补齐后，在缺失位置回填
        - 若有 mean/std，则反标准化
        返回：{sid: (n_rois, T) 的补齐矩阵}
        """
        test_data, _ = self.loader.load_dataset('test')
        results = {}

        for subj_id, subj_data in test_data.items():
            try:
                original_features = subj_data['session_1']['rest_features']   # (n_rois, T)
                missing_mask = subj_data['session_1']['missing_mask']         # (n_rois,), 1=可用, 0=缺失
                missing_mask = np.array(missing_mask).squeeze()
                if missing_mask.ndim != 1:
                    missing_mask = missing_mask.reshape(-1)

                # 统一缺失定义
                all_zero_roi = np.all(original_features == 0, axis=1)
                missing_roi = (missing_mask == 0)
                combined_missing_roi = missing_roi | all_zero_roi

                # 标准化（若有）
                if getattr(self.loader, "mean", None) is not None and getattr(self.loader, "std", None) is not None:
                    normalized = (original_features - self.loader.mean[:, np.newaxis]) / (self.loader.std[:, np.newaxis] + 1e-8)
                else:
                    normalized = original_features.copy()

                # 置零缺失处作为模型输入
                masked_features = normalized.copy()
                masked_features[combined_missing_roi, :] = 0.0

                # 构造张量
                roi_mask_2d = (missing_mask[:, np.newaxis] == 1)                     # True=可用
                test_mask = torch.tensor(roi_mask_2d, dtype=torch.bool).unsqueeze(0).to(device)
                features_tensor = torch.tensor(masked_features, dtype=torch.float32).unsqueeze(0).to(device)

                cond_np = self.loader.phenotypes.get(subj_id, np.zeros(self.cfg.cond_dim))
                cond = torch.tensor(cond_np, dtype=torch.float32).unsqueeze(0).to(device)

                # 推理
                with torch.no_grad():
                    completed_tensor = self.model(features_tensor, cond=cond, mask=test_mask)
                completed_np = completed_tensor.squeeze().cpu().numpy()               # (n_rois, T)

                # 反标准化（若有）
                if getattr(self.loader, "mean", None) is not None and getattr(self.loader, "std", None) is not None:
                    completed_np = completed_np * (self.loader.std[:, np.newaxis] + 1e-8) + self.loader.mean[:, np.newaxis]

                # 仅在缺失处回填
                final_result = original_features.copy()
                final_result[combined_missing_roi, :] = completed_np[combined_missing_roi, :]

                results[subj_id] = final_result

            except Exception as e:
                logger.error(f"测试样本 {subj_id} 处理失败: {str(e)}")

        return results
