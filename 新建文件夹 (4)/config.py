# config.py
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import random

# -----------------------------
# 路径与超参数（可按需修改）
# -----------------------------
@dataclass
class Paths:
    # 原脚本里使用的 ABIDEII 表型与时间序列目录（Windows 示例路径）
    phenotypic_csv: Path = Path(r"E:/Users/数据集/对抗网络补齐数据/ABIDEII.csv")
    ts_dir_aal: Path = Path(r"E:/Users/数据集/对抗网络补齐数据/新建文件夹")
    ts_dir_sch: Path = Path(r"E:/Users/数据集/对抗网络补齐数据/新建文件夹 (2)")
    out_dir: Path = Path("./outputs")  # 训练过程输出（日志/图/模型）

@dataclass
class GraphCfg:
    pca_components: int = 120
    k: int = 10
    similarity_threshold: float = 0.0
    z_thresh: float = 0.0   # 兼容老变量名

@dataclass
class TrainCfg:
    init_lr: float = 1e-3
    weight_decay: float = 5e-4
    dropout: float = 0.2
    batch_size: int = 64
    local_epochs: int = 2
    num_rounds: int = 200
    seed: int = 42

@dataclass
class FedCfg:
    num_clients: int = 10
    n_splits: int = 20  # StratifiedKFold 折数

@dataclass
class RunCfg:
    selected_subjects: Optional[List[str]] = None  # 若只训练指定受试者，填列表；否则为 None
    log_file: str = "model_training.log"
    log_level: int = logging.WARNING  # 与原脚本一致：WARNING

# -----------------------------
# 工具：设备、随机种子、日志
# -----------------------------
def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 让 CUDNN 可复现（如需最高性能可关闭）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dirs(paths: Paths) -> None:
    paths.out_dir.mkdir(parents=True, exist_ok=True)

def setup_logging(run: RunCfg, make_console: bool = True) -> None:
    """与原脚本风格一致的日志设置（文件 + 控制台），避免重复 handler。"""
    root = logging.getLogger("")
    # 清空已有 handlers，防止重复输出
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(run.log_level)
    fmt = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")

    fh = logging.FileHandler(run.log_file, encoding="utf-8")
    fh.setLevel(run.log_level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    if make_console:
        ch = logging.StreamHandler()
        ch.setLevel(run.log_level)
        ch.setFormatter(fmt)
        root.addHandler(ch)

# -----------------------------
# 初始化并导出全局配置（供其它模块 import）
# -----------------------------
PATHS = Paths()
GRAPH = GraphCfg()
TRAIN = TrainCfg()
FED = FedCfg()
RUN = RunCfg()

# 创建输出目录、设置日志、设定随机种子与设备
ensure_dirs(PATHS)
setup_logging(RUN, make_console=True)
seed_everything(TRAIN.seed)
device = get_device()

# -----------------------------
# 兼容原脚本的“常量名”导出
# -----------------------------
PHENOTYPIC_CSV_PATH = PATHS.phenotypic_csv
TS_DIR_AAL = PATHS.ts_dir_aal
TS_DIR_SCH = PATHS.ts_dir_sch

PCA_COMPONENTS = GRAPH.pca_components
K = GRAPH.k
SIMILARITY_THRESHOLD = GRAPH.similarity_threshold
z_thresh = GRAPH.z_thresh  # 兼容

INIT_LR = TRAIN.init_lr
WEIGHT_DECAY = TRAIN.weight_decay
DROPOUT_RATE = TRAIN.dropout
LOCAL_EPOCHS = TRAIN.local_epochs
NUM_ROUNDS = TRAIN.num_rounds

NUM_CLIENTS = FED.num_clients
N_SPLITS = FED.n_splits

SELECTED_SUBJECTS = RUN.selected_subjects  # 兼容
