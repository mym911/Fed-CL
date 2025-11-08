# main.py
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

from config import ModelConfig, DatasetConfig
from utils import logger, set_seed, compute_global_normalization_params, device
from data import ABIDEDataLoader, ABIDEDataset, collate_fn_with_dynamic_padding
from train import BrainCompletionTrainer
from test import BrainCompletionTester


def run_train_cv():
    set_seed(42)
    data_cfg = DatasetConfig()
    model_cfg = ModelConfig()

    # 1) 加载训练数据索引与表型
    loader = ABIDEDataLoader(data_cfg)
    train_dataset, _ = loader.load_dataset('train')
    if len(train_dataset) == 0:
        logger.error("训练集中没有样本，程序退出")
        return

    subject_ids = list(train_dataset.keys())
    num_rois = len(loader.custom_labels)
    logger.info(f"共加载 {len(subject_ids)} 个受试者，每个样本 {num_rois} 个 ROI")

    # 2) 全局归一化参数（逐 ROI）
    global_mean, global_std = compute_global_normalization_params(train_dataset, num_rois)
    loader.mean = global_mean
    loader.std = global_std

    # 3) 5 折交叉验证训练
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_r2_list = []
    best_r2 = -np.inf
    data_cfg.model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = data_cfg.model_dir / "roi_completion_best.pth"

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(subject_ids), start=1):
        logger.info(f"===== Fold {fold_idx} / 5 =====")

        train_ids = [subject_ids[i] for i in train_idx]
        val_ids = [subject_ids[i] for i in val_idx]
        train_fold = {sid: train_dataset[sid] for sid in train_ids}
        val_fold = {sid: train_dataset[sid] for sid in val_ids}

        # DataLoader
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            ABIDEDataset(train_fold, loader.phenotypes),
            batch_size=model_cfg.batch_size,
            collate_fn=collate_fn_with_dynamic_padding,
            shuffle=True
        )

        # Trainer
        trainer = BrainCompletionTrainer(model_cfg, data_cfg, train_loader)
        trainer.loader.mean = global_mean
        trainer.loader.std = global_std

        # 三阶段：重建 → 对抗预训练 → 对抗训练
        trainer.train()
        roi_train_data = trainer._prepare_roi_data(train_fold)
        if not roi_train_data:
            logger.warning(f"Fold {fold_idx}: 没有有效训练样本，跳过")
            continue
        trainer.pretrain(roi_train_data)
        trainer.adversarial_train(roi_train_data)

        # 保存该折模型
        fold_model_path = data_cfg.model_dir / f"roi_completion_fold{fold_idx}.pth"
        torch.save(trainer.gen.state_dict(), fold_model_path)
        logger.info(f"Fold {fold_idx} 模型已保存至 {fold_model_path}")

        # 在验证集上评估 R²
        r2_list = []
        val_data_prepped = trainer._prepare_roi_data(val_fold)
        for feats, roi_mask, cond, _ in val_data_prepped:
            sim_mask = trainer.masker.generate_mask(feats.shape, mode='roi').to(device)
            masked_input = feats.clone()
            masked_input[sim_mask] = 0.0
            with torch.no_grad():
                pred = trainer.gen(masked_input, cond=cond.unsqueeze(0), mask=(roi_mask & (~sim_mask)))
            pred_np = pred.squeeze(0).cpu().numpy()
            feats_np = feats.squeeze(0).cpu().numpy()
            sim_mask_np = sim_mask.squeeze(0).cpu().numpy()

            ss_res = np.sum((feats_np[sim_mask_np] - pred_np[sim_mask_np]) ** 2)
            ss_tot = np.sum((feats_np - feats_np.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            r2_list.append(r2)

        avg_r2 = float(np.nanmean(r2_list)) if r2_list else float("nan")
        fold_r2_list.append(avg_r2)
        logger.info(f"Fold {fold_idx} 平均 R² = {avg_r2:.4f}")

        if np.isfinite(avg_r2) and avg_r2 > best_r2:
            best_r2 = avg_r2
            torch.save(trainer.gen.state_dict(), best_model_path)
            logger.info(f"更新最佳模型 (Fold {fold_idx})，R²={avg_r2:.4f}")

    # 4) 保存归一化参数给测试阶段使用
    np.savez(data_cfg.model_dir / "norm_params.npz", mean=global_mean, std=global_std)

    # 5) 绘制 R² 曲线
    if fold_r2_list:
        try:
            import matplotlib.pyplot as plt
            x = np.arange(1, len(fold_r2_list) + 1)
            y = np.array(fold_r2_list)
            plt.figure(figsize=(6, 4))
            plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=6)
            plt.xticks(x)
            plt.ylim(0, 1)
            plt.xlabel("Fold Index")
            plt.ylabel("R\u00B2")
            plt.title("5-Fold CV Reconstruction R\u00B2")
            plt.grid(linestyle='--', alpha=0.4)
            plt.tight_layout()
            plot_dir = Path(r"E:/Users/数据集/对抗图")
            plot_dir.mkdir(parents=True, exist_ok=True)
            save_path = plot_dir / "crossval_r2_curve.png"
            plt.savefig(save_path, dpi=300)
            logger.info(f"R² 曲线图保存到 {save_path}")
        except Exception as e:
            logger.warning(f"绘图失败（可忽略）：{e}")

    logger.info(f"Cross-Validation 完成，平均 R² = {np.nanmean(np.array(fold_r2_list, dtype=float)):.4f}")


def run_test():
    set_seed(42)
    data_cfg = DatasetConfig()
    model_cfg = ModelConfig()

    tester = BrainCompletionTester(model_cfg, data_cfg)
    logger.info("开始对真实缺失测试集进行补齐")
    completed = tester.test_with_real_missing()

    # 输出目录（按 atlas 类型）
    if data_cfg.atlas_type == "aal":
        out_dir = Path(r"E:/Users/数据集/画图/aal0.1")
    elif data_cfg.atlas_type == "schaefer200":
        out_dir = Path(r"E:/Users/数据集/画图/schaefer")
    else:
        out_dir = Path(r"E:/Users/mym910/PycharmProjects/论文复现/12/ABIDEII-SU_2/ABIDEII-UCLA_1")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存 CSV（T×n_rois）
    import pandas as pd
    loader = tester.loader
    for subj, mat in completed.items():
        df = pd.DataFrame(mat.T, columns=loader.custom_labels)
        save_path = out_dir / f"{subj}.csv"
        df.to_csv(save_path, index=False)
        logger.info(f"Saved completion for {subj} → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test"], help="train: 5-fold 训练; test: 模型补齐测试")
    args = parser.parse_args()

    if args.mode == "train":
        run_train_cv()
    else:
        run_test()


if __name__ == "__main__":
    main()
