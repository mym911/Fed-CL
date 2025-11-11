# main.py
from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from config import CFG
from utils import set_seed, logger, device, fedavg
from client import Client
from server import Server
from models_gnn import SingleGraphModel


def split_clients(train_ids: List[str], num_clients: int) -> List[List[str]]:
    """将全局训练 id 均匀切成 num_clients 份。"""
    if len(train_ids) < num_clients:
        # 太少时允许有的客户端为空，但仍返回 num_clients 份
        pads = [[] for _ in range(num_clients - len(train_ids))]
        return [[sid] for sid in train_ids] + pads
    random.shuffle(train_ids)
    parts = np.array_split(train_ids, num_clients)
    return [list(x) for x in parts]


def weighted_mean(vals: List[float], weights: List[int]) -> float:
    total = float(sum(weights)) if sum(weights) > 0 else 1.0
    return sum(v * w for v, w in zip(vals, weights)) / total


def run_fold(fold: int, all_ids: List[str], raw_labels: np.ndarray):
    """单折联邦训练与评估。返回该折的 ACC/F1/AUC。"""
    logger.warning(f"\n========== Fold {fold} ==========")

    # ---- 1) 按被试做分层 K 折 ----
    skf = StratifiedKFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=42)
    # 取到当前 fold 的 (train_idx, test_idx)
    # 注意：此函数由主调方确保 fold 索引对应
    # 这里我们自己重新迭代，直到到达目标 fold
    cur = 1
    train_idx_global, test_idx_global = None, None
    for tr_idx, te_idx in skf.split(all_ids, raw_labels):
        if cur == fold:
            train_idx_global, test_idx_global = tr_idx, te_idx
            break
        cur += 1

    assert train_idx_global is not None and test_idx_global is not None
    train_ids = [all_ids[i] for i in train_idx_global]
    test_ids_global = [all_ids[i] for i in test_idx_global]

    # ---- 2) 划分客户端并创建 Client ----
    client_subsets = split_clients(train_ids, CFG.NUM_CLIENTS)
    clients: List[Client] = []
    for cid, ids in enumerate(client_subsets):
        c = Client(
            client_id=cid,
            raw_ids=ids,
            test_ids_global=test_ids_global,
            device_=device,
            k=CFG.K,
            lr=CFG.INIT_LR,
            batch_size=CFG.BATCH_SIZE,
            hidden_dim=128,
            num_classes=2,
            heads=4,
            dropout=0.5,
        )
        clients.append(c)
    logger.warning(f"[Fold{fold}] 已初始化 {len(clients)} 个客户端")

    # ---- 3) 服务端与全局模型 ----
    # 用第一个客户端的结构初始化一个同构模型作为全局模型
    in_dim = clients[0].data.size(1)
    global_model = SingleGraphModel(in_channels=in_dim, hidden_dim=128, num_classes=2).to(device)
    server = Server(global_model, device)
    global_state: Dict[str, torch.Tensor] = copy.deepcopy(global_model.state_dict())

    best_val = -np.inf
    best_state = copy.deepcopy(global_state)

    # ---- 4) 联邦训练循环 ----
    for rnd in range(1, CFG.NUM_ROUNDS + 1):
        # 下发当前全局权重
        for c in clients:
            c.load_global(global_state)

        # ===== Phase 1：带 FedProx 的本地训练（静态图）=====
        phase1_states, sizes, val_accs = [], [], []
        # FedProx μ 做简单线性调度（可按需修改）
        mu = 0.0 + (0.1 - 0.0) * (rnd / CFG.NUM_ROUNDS)
        global_param_list = [p.clone().to(device) for p in global_state.values()]

        for c in clients:
            w1, n, v_acc, v_loss = c.local_train(
                epochs=CFG.LOCAL_EPOCHS,
                global_params=global_param_list,
                mu=mu
            )
            phase1_states.append(w1)
            sizes.append(n)
            val_accs.append(v_acc)

        weighted_val1 = weighted_mean(val_accs, sizes)

        # ===== 分簇 + 两段式聚合 =====
        cluster_models, labels, best_k, best_score = server.cluster_aggregate(
            global_state=global_state,
            client_states=phase1_states,
            sizes=sizes,
            num_clients=len(clients),
            max_k=20
        )

        # 下发各自簇内模型
        for idx, c in enumerate(clients):
            c.load_global(cluster_models[int(labels[idx])])

        # ===== Phase 2：不带 FedProx 的微调（仍在各自图上）=====
        phase2_states, val_accs2 = [], []
        for c in clients:
            w2, n2, v_acc2, v_loss2 = c.local_train(epochs=CFG.LOCAL_EPOCHS)
            phase2_states.append(w2)
            val_accs2.append(v_acc2)

        weighted_val2 = weighted_mean(val_accs2, sizes)

        # 记录最佳
        if weighted_val2 > best_val:
            best_val = weighted_val2
            best_state = copy.deepcopy(fedavg(phase2_states, sizes))

        # 聚合得到新全局
        global_state = server.fedavg(phase2_states, sizes)
        server.model.load_state_dict(global_state)

        logger.warning(
            f"Round {rnd:03d} | Phase1-val={weighted_val1:.4f} "
            f"| Phase2-val={weighted_val2:.4f} | k={best_k} (sil={best_score:.3f})"
        )

    # ---- 5) 用最佳权重做最终评估 ----
    server.model.load_state_dict(best_state)
    for c in clients:
        c.load_global(best_state)

    metrics_list, counts = [], []
    for c in clients:
        m, n_test = c.evaluate_local()
        metrics_list.append(m)
        counts.append(n_test)

    # 样本数加权 ACC / F1，AUC 用非 NaN 简单平均（也可加权）
    total_n = max(sum(counts), 1)
    acc = sum(m['acc'] * n for m, n in zip(metrics_list, counts)) / total_n
    f1 = sum(m['f1'] * n for m, n in zip(metrics_list, counts)) / total_n
    aucs = [m['auc'] for m in metrics_list if not np.isnan(m['auc'])]
    auc = float(np.mean(aucs)) if len(aucs) > 0 else float('nan')

    logger.warning(f"[Fold{fold}] FINAL Test → ACC={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    return acc, f1, auc


def main():
    set_seed(42)

    # ---- 读取表型 CSV，拿到 id 与标签 ----
    ph = pd.read_csv(CFG.PHENOTYPIC_CSV_PATH)
    ph.columns = ph.columns.str.strip()

    if not {'SUB_ID', 'DX_GROUP'}.issubset(set(ph.columns)):
        raise RuntimeError("表型 CSV 必须包含列：SUB_ID, DX_GROUP")

    all_ids = ph['SUB_ID'].astype(str).tolist()
    # ABIDE: DX_GROUP==1: Control(0), ==2: Patient(1)
    raw_labels = np.where(ph['DX_GROUP'].to_numpy() == 2, 1, 0)

    folds_results: List[Dict[str, Any]] = []

    # 若只想跑一次，可把 range 改为 [1]
    for fold in range(1, CFG.N_SPLITS + 1):
        try:
            acc, f1, auc = run_fold(fold, all_ids, raw_labels)
            folds_results.append({'acc': acc, 'f1': f1, 'auc': auc})
        except Exception as e:
            logger.exception(f"[Fold{fold}] 发生异常：{e}")
            folds_results.append({'acc': 0.0, 'f1': 0.0, 'auc': float('nan')})

    ACCs = [d['acc'] for d in folds_results]
    F1s  = [d['f1'] for d in folds_results]
    AUCs = [d['auc'] for d in folds_results]

    print("\n===== Cross-Validation Summary =====")
    print(f"ACC mean ± std: {np.nanmean(ACCs):.4f} ± {np.nanstd(ACCs):.4f}")
    print(f"F1  mean ± std: {np.nanmean(F1s):.4f} ± {np.nanstd(F1s):.4f}")
    print(f"AUC mean ± std: {np.nanmean(AUCs):.4f} ± {np.nanstd(AUCs):.4f}")

    # 可选：保存结果
    out_dir = Path("./runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(folds_results).to_csv(out_dir / "cv_results.csv", index=False)
    logger.warning(f"结果已保存至 {out_dir / 'cv_results.csv'}")


if __name__ == "__main__":
    main()
