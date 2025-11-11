# models_gnn.py
from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class SingleGraphModel(nn.Module):
    """
    轻量级单层 GATv2 图模型（支持 edge_weight 经 edge_dim=1 作为边特征）。
    - 先用线性层把输入投影到 hidden_dim
    - 1 层 GATv2Conv（带残差）
    - MLP 分类头
    forward(x, edge_index, edge_weight) -> (node_emb, logits)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_classes: int,
        heads: int = 4,
        dropout: float = 0.5,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.pre = nn.Linear(in_channels, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()

        # edge_dim=1 以接收一维的 edge_weight 作为边特征
        self.gat = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            edge_dim=1
        )

        self.cls = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        # 仅对线性层做简单初始化（GAT 内部有默认初始化）
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [N, in_channels]
            edge_index: [2, E]
            edge_weight: [E] 或 None（若给定则作为边特征）

        Returns:
            h: [N, hidden_dim] 节点表征
            logits: [N, num_classes]
        """
        h0 = self.norm(self.pre(x))

        if edge_index is not None:
            edge_attr = edge_weight.unsqueeze(-1) if edge_weight is not None else None
            h1 = F.elu(self.gat(h0, edge_index, edge_attr))
            h = h0 + h1  # 残差
        else:
            h = h0

        logits = self.cls(h)
        return h, logits


class SingleGraphModelDeep(nn.Module):
    """
    稍深一些的两层 GATv2（可选，用于替换 SingleGraphModel）。
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_classes: int,
        heads: int = 4,
        dropout: float = 0.5,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.pre = nn.Linear(in_channels, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()

        self.gat1 = GATv2Conv(
            hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, edge_dim=1
        )
        self.gat2 = GATv2Conv(
            hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, edge_dim=1
        )

        self.cls = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = self.norm(self.pre(x))
        if edge_index is not None:
            ea = edge_weight.unsqueeze(-1) if edge_weight is not None else None
            h1 = F.elu(self.gat1(h0, edge_index, ea))
            h2 = F.elu(self.gat2(h1, edge_index, ea))
            h = h0 + h2  # 残差
        else:
            h = h0
        logits = self.cls(h)
        return h, logits


def build_model(
    in_channels: int,
    num_classes: int,
    hidden_dim: int = 128,
    heads: int = 4,
    dropout: float = 0.5,
    deep: bool = False,
    use_layernorm: bool = False,
) -> nn.Module:
    """
    工厂函数：按需创建单层或双层 GATv2 模型。
    """
    if deep:
        return SingleGraphModelDeep(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            heads=heads,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )
    else:
        return SingleGraphModel(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            heads=heads,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )
