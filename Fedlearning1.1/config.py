# config.py
from pathlib import Path

class CFG:
    # ====== 数据路径（按你的机器修改）======
    PHENOTYPIC_CSV_PATH = Path(r"E:/Users/数据集/对抗网络补齐数据/ABIDEII.csv")
    TS_DIR_AAL = Path(r"E:/Users/数据集/对抗网络补齐数据/新建文件夹")
    TS_DIR_SCH = Path(r"E:/Users/数据集/对抗网络补齐数据/新建文件夹 (2)")

    # ====== 模型与训练超参 ======
    PCA_COMPONENTS = 120       # PCA 维度（如未用可忽略）
    INIT_LR = 1e-3             # 初始学习率
    WEIGHT_DECAY = 5e-4        # 权重衰减
    K = 10                     # KNN 边数
    NUM_ROUNDS = 200           # 联邦训练轮数
    LOCAL_EPOCHS = 2           # 每轮本地训练 epoch 数
    NUM_CLIENTS = 10           # 客户端数量
    N_SPLITS = 20              # 交叉验证折数
    BATCH_SIZE = 64            # 本地批大小
