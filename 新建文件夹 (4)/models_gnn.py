# models_gnn.py
from __future__ import annotations

from typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


__all__ = [
    "SingleGraphModel",
    "TwoLayerGAT",
    "GraphMLP",
    "count_parameters",
    "build_single_graph_model",
]


class SingleGraphModel(nn.Module):
    """
    轻量 GATv2 节点分类模型（支持边权 edge_weight，经由 edge_dim=1 输入到 GATv2Conv）

    输入:
      - x:           [N, in_channels]
      - edge_index:  [2, E]
      - edge_weight: [E]  (会在内部 unsqueeze(-1) 变成 [E, 1])

    输出:
      - h:       [N, hidden_dim]  节点嵌入
      - logits:  [N, num_classes] 分类 logits
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_classes: int,
        heads: int = 4,
        dropout: float = 0.5,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        pre_layers = [nn.Linear(in_channels, hidden_dim, bias=False)]
        if use_layernorm:
            pre_layers += [nn.LayerNorm(hidden_dim)]
        pre_layers += [nn.ReLU(inplace=True), nn.Dropout(p=0.1)]
        self.pre = nn.Sequential(*pre_layers)

        # 单层 GATv2，支持边特征（edge_dim=1）
        self.gat = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            edge_dim=1,  # 把标量边权作为 1 维边特征
            add_self_loops=False,
        )

        # 节点级分类头
        self.cls = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
          h:       节点嵌入
          logits:  分类 logits
        """
        h0 = self.pre(x)

        if edge_index is not None:
            edge_attr = edge_weight.unsqueeze(-1) if edge_weight is not None else None
            h1 = F.elu(self.gat(h0, edge_index, edge_attr))
            h = h0 + h1  # 残差
        else:
            h = h0

        logits = self.cls(h)
        return h, logits


class TwoLayerGAT(nn.Module):
    """
    两层 GATv2 变体：更强表达力，可在数据量更大时尝试
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_classes: int,
        heads: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(in_channels, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.gat1 = GATv2Conv(
            hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, edge_dim=1, add_self_loops=False
        )
        self.gat2 = GATv2Conv(
            hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, edge_dim=1, add_self_loops=False
        )
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.pre(x)
        if edge_index is not None:
            ea = edge_weight.unsqueeze(-1) if edge_weight is not None else None
            h1 = F.elu(self.gat1(h, edge_index, ea))
            h1 = self.dropout(h1)
            h2 = F.elu(self.gat2(h1, edge_index, ea))
            h = h + h2  # 残差到输入
        logits = self.cls(h)
        return h, logits


class GraphMLP(nn.Module):
    """
    不使用图卷积的 MLP 基线（用于 ablation）
    """

    def __init__(self, in_channels: int, hidden_dim: int, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.cls = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.mlp(x)
        logits = self.cls(h)
        return h, logits


# --------- helpers ---------
def count_parameters(model: nn.Module) -> int:
    """返回可训练参数个数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_single_graph_model(
    in_dim: int,
    num_classes: int,
    hidden_dim: int = 128,
    heads: int = 4,
    dropout: float = 0.5,
) -> SingleGraphModel:
    """
    简洁工厂函数，与你的其余模块对接方便：
      model = build_single_graph_model(in_dim, num_classes)
    """
    return SingleGraphModel(
        in_channels=in_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        heads=heads,
        dropout=dropout,
    )
