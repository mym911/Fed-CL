# models.py
import torch
import torch.nn as nn

class ROICompletionGenerator(nn.Module):
    """
    条件式生成器：输入被掩蔽的 ROI×T 时间序列与表型 cond，
    通过 1D 卷积编码/解码补齐缺失 ROI。
    """
    def __init__(self, config, num_rois: int):
        super().__init__()
        self.num_rois = num_rois
        self.cond_embed = nn.Linear(config.cond_dim, num_rois)  # 将 [age, sex] 映射到每个 ROI 的偏置
        self.encoder = nn.Sequential(
            nn.Conv1d(num_rois, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1)
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, num_rois, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor):
        """
        x:    [B, n_rois, T]（输入序列，缺失处通常置 0）
        cond: [B, cond_dim]
        mask: [B, n_rois, T]（True=可用/参与重建的区域）
        """
        cond_emb = self.cond_embed(cond).unsqueeze(-1).expand(-1, -1, x.shape[-1])  # [B, n_rois, T]
        x_masked = x * mask.float()
        x_enriched = x_masked + cond_emb * mask.float()
        z = self.encoder(x_enriched)
        out = self.decoder(z)
        return out


class ROICompletionDiscriminator(nn.Module):
    """
    判别器：判断输入 ROI×T 是否为“真实/补齐”。
    仅在 mask 指示的可用位置聚合特征。
    """
    def __init__(self, num_rois: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(num_rois, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        x:    [B, n_rois, T]
        mask: [B, n_rois, T]（True=可用）
        """
        x = x * mask.float()
        h = self.conv(x)
        h = self.pool(h).flatten(1)  # [B, 64]
        return self.head(h)
