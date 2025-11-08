# train.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict

from utils import device, logger, DynamicMaskGenerator
from models import ROICompletionGenerator, ROICompletionDiscriminator
from data import ABIDEDataset, ABIDEDataLoader, collate_fn_with_dynamic_padding

class BrainCompletionTrainer:
    def __init__(self, model_config, data_config, train_data_loader: DataLoader):
        self.cfg = model_config
        self.data_cfg = data_config
        self.loader = ABIDEDataLoader(data_config)
        self.train_data_loader = train_data_loader

        num_rois = len(self.loader.custom_labels)
        self.masker = DynamicMaskGenerator(mask_rate=model_config.mask_rate)

        self.gen  = ROICompletionGenerator(model_config, num_rois=num_rois).to(device)
        self.disc = ROICompletionDiscriminator(num_rois=num_rois).to(device)

        self.opt_g = optim.AdamW(self.gen.parameters(), lr=model_config.lr)
        self.opt_d = optim.AdamW(self.disc.parameters(), lr=model_config.lr * 0.5)

        self.recon_loss = nn.L1Loss(reduction='none')
        self.adv_loss   = nn.BCELoss()

    def train(self):
        """重建预训练（仅重建损失）"""
        logger.info("开始训练(初步重建)...")
        for epoch in range(self.cfg.epochs_pretrain):
            total_loss = 0.0
            for batch_idx, (features, roi_avail_mask, cond) in enumerate(self.train_data_loader):
                simulated_missing_mask = self.masker.generate_mask(features.shape, mode='roi').to(device)
                masked_input = features.clone()
                masked_input[simulated_missing_mask] = 0.0
                valid_mask = roi_avail_mask & (~simulated_missing_mask)

                pred = self.gen(masked_input, cond=cond, mask=valid_mask)

                time_padding_mask   = (features != 0).any(dim=1, keepdim=True)
                overall_valid_mask  = valid_mask & time_padding_mask
                pred_valid   = pred[overall_valid_mask]
                target_valid = features[overall_valid_mask]

                loss = self.recon_loss(pred_valid, target_valid).mean()

                self.opt_g.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.gen.parameters(), 1.0)
                self.opt_g.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.cfg.epochs_pretrain}, "
                            f"Reconstruction Loss={total_loss / (batch_idx+1):.4f}")

    def _prepare_roi_data(self, data: Dict) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """把字典数据打包成(特征,mask,条件,可用掩码向量)列表，方便对抗训练阶段逐样本迭代。"""
        roi_data = []
        for subj_id, subj_data in data.items():
            if 'session_1' in subj_data and 'rest_features' in subj_data['session_1']:
                try:
                    features = subj_data['session_1']['rest_features']
                    roi_available_mask = subj_data['session_1']['missing_mask']  # (n_rois,)
                    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

                    T = features.shape[-1]
                    import numpy as np
                    mask_2d = (np.tile(roi_available_mask[:, np.newaxis], (1, T)) == 1)
                    mask_tensor = torch.tensor(mask_2d, dtype=torch.bool).unsqueeze(0).to(device)

                    cond_np = self.loader.phenotypes.get(subj_id, np.zeros(self.cfg.cond_dim, dtype=np.float32))
                    cond = torch.tensor(cond_np, dtype=torch.float32).to(device)

                    roi_data.append((features_tensor, mask_tensor, cond, roi_available_mask))
                except Exception as e:
                    logger.error(f"处理 {subj_id} 失败: {str(e)}")
        return roi_data

    def pretrain(self, dataset):
        """对抗式预训练：先更稳地训练判别器，再训练生成器（重建+对抗）。"""
        if len(dataset) == 0:
            logger.error("训练数据集为空！")
            return

        logger.info("开始监督预训练(带GAN判别器)...")
        # --- 1) 训练判别器 ---
        self.gen.eval()
        self.disc.train()
        for features, roi_avail_mask, cond, _ in dataset:
            cond_b = cond.unsqueeze(0)
            sim_mask = self.masker.generate_mask(features.shape).to(device)
            masked_input = features * (~sim_mask).float()
            valid_mask = roi_avail_mask & (~sim_mask)

            with torch.no_grad():
                fake = self.gen(masked_input, cond=cond_b, mask=valid_mask)

            real_pred = self.disc(features, mask=roi_avail_mask)
            fake_pred = self.disc(fake,     mask=roi_avail_mask)

            real_labels = torch.ones_like(real_pred)
            fake_labels = torch.zeros_like(fake_pred)

            d_loss = 0.5*(self.adv_loss(real_pred, real_labels) +
                          self.adv_loss(fake_pred, fake_labels))

            self.opt_d.zero_grad()
            d_loss.backward()
            nn.utils.clip_grad_norm_(self.disc.parameters(), 1.0)
            self.opt_d.step()

        # --- 2) 训练生成器 ---
        self.gen.train()
        self.disc.eval()
        for features, roi_avail_mask, cond, _ in dataset:
            cond_b = cond.unsqueeze(0)
            sim_mask = self.masker.generate_mask(features.shape).to(device)
            masked_input = features * (~sim_mask).float()
            valid_mask = roi_avail_mask & (~sim_mask)

            fake = self.gen(masked_input, cond=cond_b, mask=valid_mask)
            gen_pred = self.disc(fake, mask=roi_avail_mask)
            g_loss_adv = self.adv_loss(gen_pred, torch.ones_like(gen_pred))

            time_padding_mask = (features != 0).any(dim=1, keepdim=True)
            final_valid_mask = valid_mask & time_padding_mask
            g_loss_rec = self.recon_loss(fake[final_valid_mask], features[final_valid_mask]).mean()

            g_loss = 0.1*g_loss_adv + 0.9*g_loss_rec

            self.opt_g.zero_grad()
            g_loss.backward()
            nn.utils.clip_grad_norm_(self.gen.parameters(), 1.0)
            self.opt_g.step()

    def adversarial_train(self, dataset):
        """完整对抗训练循环（交替优化D/G）。"""
        logger.info("开始对抗训练...")
        grad_clip = nn.utils.clip_grad_norm_

        for epoch in range(self.cfg.epochs_gan):
            d_loss_total = 0.0
            g_loss_total = 0.0

            for features, roi_avail_mask, cond, _ in dataset:
                batch_size = features.size(0)

                # --- 训练判别器 D ---
                self.disc.train(); self.gen.eval()
                sim_mask = self.masker.generate_mask(features.shape, mode='roi').to(device)
                masked_input = features.clone()
                masked_input[sim_mask] = 0.0
                valid_mask = roi_avail_mask & (~sim_mask)
                cond_b = cond.unsqueeze(0)

                with torch.no_grad():
                    fake = self.gen(masked_input, cond=cond_b, mask=valid_mask)

                real_pred = self.disc(features,     mask=valid_mask)
                fake_pred = self.disc(fake.detach(), mask=valid_mask)

                real_labels = torch.ones(batch_size, 1, device=device).uniform_(0.9, 1.0)
                fake_labels = torch.zeros(batch_size, 1, device=device).uniform_(0.0, 0.1)

                d_loss = 0.5*(self.adv_loss(real_pred, real_labels) +
                              self.adv_loss(fake_pred, fake_labels))

                self.opt_d.zero_grad()
                d_loss.backward()
                grad_clip(self.disc.parameters(), 1.0)
                self.opt_d.step()

                # --- 训练生成器 G ---
                self.disc.eval(); self.gen.train()
                fake = self.gen(masked_input, cond=cond_b, mask=valid_mask)
                gen_pred = self.disc(fake, mask=valid_mask)
                g_loss_adv = self.adv_loss(gen_pred, real_labels)

                time_padding_mask = (features != 0).any(dim=1, keepdim=True)
                final_valid_mask = valid_mask & time_padding_mask
                g_loss_rec = self.recon_loss(fake[final_valid_mask], features[final_valid_mask]).mean()

                g_loss = 0.05*g_loss_adv + 0.95*g_loss_rec

                self.opt_g.zero_grad()
                g_loss.backward()
                grad_clip(self.gen.parameters(), 1.0)
                self.opt_g.step()

                d_loss_total += d_loss.item()
                g_loss_total += g_loss.item()

            if (epoch + 1) % 10 == 0:
                n = max(len(dataset), 1)
                logger.info(f"对抗训练 Epoch {epoch+1}/{self.cfg.epochs_gan} "
                            f"| D Loss: {d_loss_total/n:.4f} | G Loss: {g_loss_total/n:.4f}")
