# transformers.py
from __future__ import annotations

from typing import Optional, Sequence
import torch
from torch import nn
from torch.nn import functional as F


__all__ = [
    "AttentionLayer",
    "Attention_LastDim",
    "WindowFusionTransformer",
    "WindowFusionTransformerSimple",
]


class AttentionLayer(nn.Module):
    """
    标准多头自注意力（不依赖 torch.nn.MultiheadAttention 的轻量实现）。
    输入:  x [B, T, C]
    输出:  y [B, T, C], attn [B, heads, T, T]
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop_ratio: float = 0.0,
        proj_drop_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                      # [B, H, T, C//H]
        attn = (q @ k.transpose(-2, -1)) * self.scale         # [B, H, T, T]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        y = (attn @ v).transpose(1, 2).reshape(B, T, C)       # [B, T, C]
        y = self.proj(y)
        y = self.proj_drop(y)
        return y, attn


class Attention_LastDim(nn.Module):
    """
    特征维注意力（对 [B, C] 进行通道注意力），可选对先验索引增强。
    输入:  x [B, C]
    输出:  注意力权重 [B, C, 1]（便于与 [B, C] 或 [B, C, 1] 相乘）
    """
    def __init__(
        self,
        input_dim: int,
        hid_dim: int,
        prior_indices: Optional[Sequence[int]] = None,
        prior_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.attn_layer = nn.Sequential(
            nn.Linear(input_dim, hid_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, input_dim),
            nn.Sigmoid(),
        )
        self.prior_indices = list(prior_indices) if prior_indices is not None else None
        self.prior_weight = float(prior_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C]
        w = self.attn_layer(x)  # [B, C]
        if self.prior_indices:
            prior = torch.zeros_like(w)
            prior[:, self.prior_indices] = self.prior_weight
            w = torch.clamp(w + prior, 0.0, 1.0)
        return w.unsqueeze(-1)   # [B, C, 1]


class WindowFusionTransformer(nn.Module):
    """
    改进版窗口级 Transformer 融合：
      1) 线性投影到 d_model
      2) 可学习位置编码（广播）
      3) TransformerEncoder(可多层)
      4) 时序平均池化
      5) 特征维注意力加权
      6) 最终融合头

    输入:  x [B, T, D_in]
    输出:  fused [B, d_model]
    """
    def __init__(
        self,
        input_dim: int,
        fused_dim: int = 512,         # d_model
        num_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_custom_attn: bool = False # 如需在 encoder 后叠一层自定义 AttentionLayer
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, fused_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 1, fused_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fused_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.use_custom_attn = use_custom_attn
        if use_custom_attn:
            self.custom_attn = AttentionLayer(fused_dim, num_heads=n_heads)

        self.attn_lastdim = Attention_LastDim(fused_dim, fused_dim // 2)
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fused_dim // 2, fused_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, Din] -> [B, T, d_model]
        x = self.input_proj(x)
        x = x + self.pos_embed
        x = self.encoder(x)
        if self.use_custom_attn:
            x, _ = self.custom_attn(x)
        # Time pooling
        pooled = x.mean(dim=1)                  # [B, d_model]
        w = self.attn_lastdim(pooled).squeeze(-1)  # [B, d_model]
        fused = pooled * w
        return self.fusion(fused)               # [B, d_model]


class WindowFusionTransformerSimple(nn.Module):
    """
    精简版窗口级 Transformer 融合（与原始单文件代码接口一致）。
    输入:  x [B, T, D_in]
    输出:  out [B, fused_dim]
    """
    def __init__(
        self,
        input_dim: int,
        fused_dim: int = 128,
        num_layers: int = 1,
        n_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, fused_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, fused_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fused_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attn_lastdim = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fused_dim // 2, fused_dim),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, Din] -> [B, T, fused_dim]
        x = self.input_proj(x)
        x = x + self.pos_embed
        x = self.encoder(x)
        pooled = x.mean(dim=1)                 # [B, fused_dim]
        w = self.attn_lastdim(pooled)          # [B, fused_dim]
        out = pooled * w
        return self.fusion(out)                # [B, fused_dim]
