# transformers.py
import torch
from torch import nn

class AttentionLayer(nn.Module):
    """标准多头自注意力（用于可选的细化特征交互）"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop_ratio=0., proj_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # x: [B, T, C]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)\
                         .permute(2, 0, 3, 1, 4)  # [3, B, H, T, C//H]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn


class Attention_LastDim(nn.Module):
    """对最后一维（特征维）做通道注意力，用于加权融合后的特征"""
    def __init__(self, input_dim, hid_dim):
        super().__init__()
        self.attn_layer = nn.Sequential(
            nn.Linear(input_dim, hid_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hid_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C]
        return self.attn_layer(x).unsqueeze(-1)  # [B, C, 1]


class WindowFusionTransformerSimple(nn.Module):
    """
    精简版时间窗口融合 Transformer
    输入:  x 形状 [B, T, D_in]
    输出:  [B, F]  (F = fused_dim)
    """
    def __init__(self, input_dim: int, fused_dim: int = 128,
                 num_layers: int = 2, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # 1) 输入线性投影：将 D_in -> fused_dim
        self.input_proj = nn.Linear(input_dim, fused_dim)

        # 2) 简易可学习“位置编码”（广播相加）
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, fused_dim))

        # 3) Transformer 编码器
        enc_layer = nn.TransformerEncoderLayer(
            d_model=fused_dim, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # 4) 特征级注意力
        self.attn_lastdim = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fused_dim // 2, fused_dim),
            nn.Sigmoid()
        )

        # 5) 融合头
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D_in]
        x = self.input_proj(x)    # [B, T, F]
        x = x + self.pos_embed    # 位置编码（广播）
        x = self.encoder(x)       # [B, T, F]

        pooled = x.mean(dim=1)    # 时间平均池化 → [B, F]
        attn_w = self.attn_lastdim(pooled)   # [B, F]
        out = self.fusion(pooled * attn_w)   # [B, F]
        return out
