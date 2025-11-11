# utils.py
import os
import random
import logging
import warnings
from collections import OrderedDict
from typing import Dict

import numpy as np
import torch

# ===== 日志与告警设置 =====
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)

logging.basicConfig(
    filename='model_training.log',
    level=logging.WARNING,
    format='%(asctime)s:%(levelname)s:%(message)s'
)
_console = logging.StreamHandler()
_console.setLevel(logging.WARNING)
_console.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(message)s'))
logging.getLogger('').addHandler(_console)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ===== 设备 =====
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int = 42):
    """固定随机种子，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def flatten_state_dict(state: Dict[str, torch.Tensor]) -> np.ndarray:
    """将 state_dict 参数按 key 排序并展平为 1D numpy 向量。"""
    parts = []
    for k in sorted(state.keys()):
        parts.append(state[k].detach().cpu().numpy().ravel())
    return np.concatenate(parts, axis=0)


class FocalLoss(torch.nn.Module):
    """多分类 Focal Loss（默认 gamma=2）。"""
    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def fedavg(states, sizes):
    """Federated Averaging 聚合。"""
    total = sum(sizes)
    agg = OrderedDict()
    for k in states[0]:
        agg[k] = sum(states[i][k] * (sizes[i] / total) for i in range(len(states)))
    return agg
