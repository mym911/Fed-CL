# main.py
from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from config import (
    PHENOTYPIC_CSV_PATH,
    NUM_CLIENTS,
    NUM_ROUNDS,
    LOCAL_EPOCHS,
    N_SPLITS,
    SEED,
)
from utils import set_seed
from client import Client
from server import Server


def weighted_mean(values: List[float], weights: List[int]) -> float:
    total = float(sum(weights)) if sum(weights) > 0 else 1.0
    return sum(v * w for v, w in zip(values, weights)) / total


def run_fold(
    fold_idx: int,
    train_ids: List[str],
    test_ids_global: List[str],
    device: torch.device,
) -> Dict[str, float]:
    """
    单折联邦训练 + 测试
    """
    # ---- 将训练 IDs 均匀划给多个客户端 ----
    ids_shuffled = train_ids[:]
    random.shuffle(ids_shuffled)
    client_subsets = np.array_split(ids_shuffled, NUM_CLIENTS)

    # ---- 初始化客户端（本地预处理、建图、建模）----
    clients: List[Client] = []
    for cid, sub_ids in enumerate(client_subsets):
        clients.append(
            Client(
                client_id=cid,
                raw_ids=list(map(str, sub_ids)),
                test_ids_global=test_ids_global,
                device=device,
            )
        )

    # 以第一个客户端模型为原型，创建服务器
    server = Server(model=copy.deepcopy(clients[0].model), device=device)
    global_state = copy.deepcopy(clients[0].model.state_dict())

    best_val = -1e18

    # -----------------------------
    # 联邦训练轮次
    # -----------------------------
    for rnd in range(1, NUM_ROUNDS + 1):
        # 下发当前全局权重
        for c in clients:
            c.load_global(global_state)

        # ---------- 阶段一：带 FedProx 的本地训练（静态图） ----------
        phase1_states, sizes, val_accs = [], [], []
        # 动态设置 FedProx 系数
        mu_init, mu_max = 0.0, 0.01
        mu = mu_init + (mu_max - mu_init) * (rnd / max(1, NUM_ROUNDS))

        # 需要把全局参数展开传入本地（FedProx）
        global_param_list = [p.clone().to(device) for p in global_state.values()]

        for c in clients:
            w_local, n_nodes, v_acc, _ = c.local_train(
                epochs=LOCAL_EPOCHS,
                global_params=global_param_list,
                mu=mu,
            )
            phase1_states.append(w_local)
            sizes.append(int(n_nodes))
            val_accs.append(float(v_acc))

        weighted_val1 = weighted_mean(val_accs, sizes)

        # ---------- 服务器端：基于 Δw 的谱聚类聚合 ----------
        new_global, info = server.step(
            client_states=phase1_states,
            sizes=sizes,
            mode="cluster",
            global_state=global_state,
            max_k=min(10, NUM_CLIENTS - 1),
        )

        # 将各簇模型分别下发至对应客户端
        labels = info["labels"]
        cluster_models = info["cluster_models"]
        for idx, c in enumerate(clients):
            cid = int(labels[idx])
            c.load_global(cluster_models[cid])

        # ---------- 阶段二：簇模型初始化下的本地微调 ----------
        phase2_states, val_accs2 = [], []
        for c in clients:
            w_local, _, v_acc2, _ = c.local_train(epochs=LOCAL_EPOCHS)
            phase2_states.append(w_local)
            val_accs2.append(float(v_acc2))

        weighted_val2 = weighted_mean(val_accs2, sizes)

        # 聚合第二阶段权重，更新全局
        global_state = server.fedavg(phase2_states, sizes)
        server.load_state_dict(global_state)

        # 记录最佳
        if weighted_val2 > best_val:
            best_val = weighted_val2
            server.update_best(best_val)

        print(
            f"[Fold {fold_idx:02d}] Round {rnd:03d} | "
            f"Phase1-VAL={weighted_val1:.4f} | Phase2-VAL={weighted_val2:.4f} | "
            f"Clusters={info.get('k', '-')}, Sil={info.get('silhouette', float('nan')):.3f}"
        )

    # -----------------------------
    # 使用最佳全局权重进行测试
    # -----------------------------
    server.load_best_state()
    best_global = server.state_dict()
    for c in clients:
        c.load_global(best_global)

    # 客户端各自评估（全局测试集），再做样本数加权
    metrics, counts = [], []
    for c in clients:
        m, n = c.evaluate_local()
        metrics.append(m)
        counts.append(n if n is not None else 0)

    total_n = sum(counts) if sum(counts) > 0 else 1
    acc = sum(m["acc"] * n for m, n in zip(metrics, counts)) / total_n
    f1 = sum(m["f1"] * n for m, n in zip(metrics, counts)) / total_n
    auc = np.nanmean([m["auc"] for m in metrics])

    print(
        f"[Fold {fold_idx:02d}] FINAL Test  → "
        f"ACC={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}"
    )
    return {"acc": acc, "f1": f1, "auc": auc}


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 读取受试者与标签
    df_pheno = pd.read_csv(Path(PHENOTYPIC_CSV_PATH))
    all_ids = df_pheno["SUB_ID"].astype(str).tolist()
    raw_labels = (df_pheno["DX_GROUP"].values == 2).astype(int)  # 1: ASD, 0: TC

    # K-fold
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_results: List[Dict[str, float]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(all_ids, raw_labels), start=1):
        train_ids = [all_ids[i] for i in train_idx]
        test_ids_global = [all_ids[i] for i in test_idx]

        res = run_fold(
            fold_idx=fold_idx,
            train_ids=train_ids,
            test_ids_global=test_ids_global,
            device=device,
        )
        fold_results.append(res)

    # 汇总
    accs = [d["acc"] for d in fold_results]
    f1s = [d["f1"] for d in fold_results]
    aucs = [d["auc"] for d in fold_results]

    print("\n===== Cross-Validation Summary =====")
    print(f"ACC mean ± std: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"F1  mean ± std: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"AUC mean ± std: {np.nanmean(aucs):.4f} ± {np.nanstd(aucs):.4f}")


if __name__ == "__main__":
    main()
