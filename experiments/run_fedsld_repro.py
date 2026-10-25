"""
Chạy FL simulation giống setup của FedSLD paper (ISBI 2022).
Hỗ trợ các model: 20K (SimpleCNN) và 570K (CNN570K chuẩn bài báo).
Hỗ trợ các chiến lược: fedavg, fedprox, fedbn, fednova, scaffold, fednova_scaffold.

Chạy:
    # Chạy tất cả (2 dataset x 2 partition x các strategy)
    python -m experiments.run_fedsld_repro --all

    # Chạy cụ thể
    python -m experiments.run_fedsld_repro \
        --dataset organmnist_axial \
        --partition practical_noniid \
        --strategy fednova \
        --model 570k
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

# Ensure Matplotlib uses a writable local directory before any sub-imports
_MPL_DIR = Path("results") / ".matplotlib"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPL_DIR)

import numpy as np
import yaml
import torch
import flwr as fl

import shutil
from datasets.medmnist_code import get_pathmnist_datasets, get_organmnist_datasets
from datasets.partition import load_partition, load_partition_with_test
from models.cnn import SimpleCNN, get_parameters
from models.cnn_570k import CNN570K, CNN570K_BN, get_parameters_fedbn
from client.client import make_client_fn, make_fedbn_client_fn
from client.client_fednova import make_fednova_client_fn
from client.client_scaffold import make_scaffold_client_fn
from client.client_scaffold_la import make_scaffold_la_client_fn
from client.client_fedbn_scaffold_la import make_fedbn_scaffold_la_client_fn, reset_client_caches
from client.client_fednova_scaffold import make_fednova_scaffold_client_fn
from server.server import get_strategy, EarlyStoppingCallback

# ── Config ────────────────────────────────────────────────────────────────────
CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "experiment.yaml"
with open(CFG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)
FS = CFG["fedsld_setup"]   # fedsld_setup block

DATASET_FNS = {
    "pathmnist":        (get_pathmnist_datasets,  9),
    "organmnist_axial": (get_organmnist_datasets, 11),
}
PARTITION_DIR = Path("data/partitions")
RESULTS_DIR   = Path(FS["results_dir"])


# ── Metrics helper ────────────────────────────────────────────────────────────
class EarlyStopException(Exception):
    pass


def run_one(dataset_name, partition_method, strategy_name, device, model_type="20k", run_id=1, custom_num_rounds=None, custom_proximal_mu=None, custom_tau_la=None):
    get_fn, num_classes = DATASET_FNS[dataset_name]
    num_rounds = custom_num_rounds if custom_num_rounds is not None else FS["partition_settings"][partition_method]["fl_rounds"]
    proximal_mu_val = custom_proximal_mu if custom_proximal_mu is not None else float(FS.get("proximal_mu", 0.01))
    if custom_tau_la is not None:
        tau_la_val = custom_tau_la
    elif strategy_name in ["fedbn_scaffold_la", "fedbn_scaffold", "proposed"]:
        tau_la_val = 0.0
    else:
        tau_la_val = float(FS.get("tau_la", 1.0))

    model_label = "~570K params (FedSLD Paper)" if model_type == "570k" else "~20K params (Ours)"
    print(f"\n{'='*65}")
    print(f"  DATASET  : {dataset_name.upper()}")
    print(f"  PARTITION: {partition_method}")
    if strategy_name == "fedprox":
        print(f"  STRATEGY : {strategy_name.upper()} (proximal_mu={proximal_mu_val})")
    elif strategy_name in ["scaffold_la", "fedavg_la", "fedprox_la", "fedbn_scaffold_la", "fedbn_scaffold"]:
        print(f"  STRATEGY : {strategy_name.upper()} (tau_la={tau_la_val})")
    else:
        print(f"  STRATEGY : {strategy_name.upper()}")
    print(f"  MODEL    : {model_type.upper()} ({model_label})")
    print(f"  RUN      : {run_id}/3 (Seed 42)")
    print(f"  CLIENTS  : {FS['num_clients']} | BATCH SIZE: {FS['batch_size']}")
    print(f"  ROUNDS   : {num_rounds} | LR: {FS['learning_rate']} ({FS['optimizer'].upper()})")
    print(f"{'='*65}")

    if "fedbn" in strategy_name or strategy_name == "proposed":
        reset_client_caches()
        shutil.rmtree(Path("data/fedbn_states"), ignore_errors=True)
        Path("data/fedbn_states").mkdir(parents=True, exist_ok=True)
    if "scaffold" in strategy_name:
        shutil.rmtree(Path("data/scaffold_states"), ignore_errors=True)
        Path("data/scaffold_states").mkdir(parents=True, exist_ok=True)

    # Dataset
    train_ds, val_ds, test_ds, _ = get_fn(download=False)

    # Partition
    json_file = PARTITION_DIR / f"{partition_method}_{dataset_name}_seed42_clients12.json"
    if not json_file.exists():
        print(f"  [SKIP] Partition file không tồn tại: {json_file}")
        print(f"  -> Chạy trước: python -m datasets.partition_fedsld")
        return None
    train_partition, test_partition = load_partition_with_test(str(json_file))

    # Model
    in_channels = train_ds[0][0].shape[0]
    if model_type == "570k":
        if strategy_name in ["fedbn_scaffold_la", "fedbn_scaffold"]:
            net = CNN570K_BN(in_channels=in_channels, num_classes=num_classes).to(device)
            init_params = fl.common.ndarrays_to_parameters(get_parameters_fedbn(net))
        else:
            net = CNN570K(in_channels=in_channels, num_classes=num_classes).to(device)
            init_params = fl.common.ndarrays_to_parameters(get_parameters(net))
    else:
        net = SimpleCNN(in_channels=in_channels, num_classes=num_classes).to(device)
        init_params = fl.common.ndarrays_to_parameters(get_parameters(net))

    # Strategy
    strategy = get_strategy(strategy_name, FS["num_clients"], init_params,
                            proximal_mu=proximal_mu_val)

    # Client factory routing
    if strategy_name == "fedbn":
        client_fn_maker = make_fedbn_client_fn
    elif strategy_name == "fednova":
        client_fn_maker = make_fednova_client_fn
    elif strategy_name == "scaffold":
        client_fn_maker = make_scaffold_client_fn
    elif strategy_name == "scaffold_la":
        client_fn_maker = make_scaffold_la_client_fn
    elif strategy_name in ["fedbn_scaffold_la", "fedbn_scaffold"]:
        client_fn_maker = make_fedbn_scaffold_la_client_fn
    elif strategy_name == "fednova_scaffold":
        client_fn_maker = make_fednova_scaffold_client_fn
    else:
        client_fn_maker = make_client_fn

    # Phân bổ tài nguyên Ray: Sử dụng GPU nếu có (num_gpus=0.5 chạy 2 client song song trên GPU RTX 3050 8GB)
    if device.type == "cuda":
        gpu_res = 0.5
        client_res = {"num_cpus": 2, "num_gpus": gpu_res}
        client_device = device
    else:
        client_res = {"num_cpus": 2, "num_gpus": 0.0}
        client_device = torch.device("cpu")

    client_kwargs = dict(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        partition=train_partition,
        local_epochs=FS["local_epochs"],
        lr=FS["learning_rate"],
        device=client_device,
        num_classes=num_classes,
        batch_size=FS["batch_size"],
        optimizer_type=FS["optimizer"],
        momentum=FS.get("momentum", 0.0),
        weight_decay=FS.get("weight_decay", 0.0),
        model_type=model_type,
    )
    if strategy_name in ["scaffold_la", "fedbn_scaffold_la", "fedbn_scaffold"]:
        client_kwargs["tau_la"] = tau_la_val
        client_kwargs["test_partition"] = test_partition
    elif strategy_name in ["fedavg_la", "fedprox_la"]:
        client_kwargs["use_la"] = True
        client_kwargs["tau_la"] = tau_la_val

    client_fn = client_fn_maker(**client_kwargs)

    # Tracking metrics
    history_bta          = []   # Best Test Accuracy per round (weighted avg)
    history_bmcta        = []   # Best Mean Client Test Accuracy per round (simple mean)
    history_server_loss  = []   # Test / Evaluation Loss (Server / Combined) per round
    history_client_loss  = []   # Training Loss (Client Weighted Mean) per round
    round_times          = []
    actual_rounds        = [0]
    round_start          = [time.time()]
    es_cfg               = FS.get("early_stopping", {"enabled": False})

    early_stopper = EarlyStoppingCallback(
        patience=es_cfg.get("patience", 10),
        min_delta=es_cfg.get("min_delta", 0.001)
    ) if es_cfg.get("enabled", False) else None

    # 1. Bọc strategy.aggregate_fit để thu thập Client Training Loss trung bình từng round
    orig_agg_fit = strategy.aggregate_fit

    def agg_fit_with_tracking(server_round, results, failures):
        agg_res = orig_agg_fit(server_round, results, failures)
        if results:
            total_train_samples = sum(fit_res.num_examples for _, fit_res in results)
            weighted_train_loss = sum(
                fit_res.num_examples * float(fit_res.metrics.get("train_loss", 0.0))
                for _, fit_res in results
                if "train_loss" in fit_res.metrics
            )
            round_train_loss = (weighted_train_loss / total_train_samples) if total_train_samples > 0 else 0.0
            history_client_loss.append(round(float(round_train_loss), 4))
        return agg_res

    strategy.aggregate_fit = agg_fit_with_tracking

    # 2. Bọc strategy.aggregate_evaluate để thu thập Server Test Loss và Accuracy
    orig_agg = strategy.aggregate_evaluate

    def agg_with_tracking(server_round, results, failures):
        t = time.time() - round_start[0]
        round_times.append(t)
        round_start[0] = time.time()
        actual_rounds[0] = server_round

        agg = orig_agg(server_round, results, failures)
        if agg is not None and results:
            loss_agg, metrics_agg = agg

            # Server / Test Loss
            if loss_agg is not None:
                history_server_loss.append(round(float(loss_agg), 4))
            else:
                history_server_loss.append(0.0)

            # BTA:
            # Với các chiến lược FedBN: mỗi client duy trì thống kê BatchNorm cá nhân hóa.
            # Do đó BTA được đo bằng Combined Accuracy (weighted average) trên các partition test cục bộ của 12 client,
            # bao phủ 100% test set mà không làm sai lệch thống kê running mean/var của BN.
            is_fedbn = ("fedbn" in strategy_name) or (strategy_name == "proposed")
            if is_fedbn:
                bta = metrics_agg.get("accuracy", 0.0)
            elif "global_accuracy" in metrics_agg:
                bta = metrics_agg.get("global_accuracy", 0.0)
            else:
                bta = metrics_agg.get("accuracy", 0.0)
            history_bta.append(bta)

            # BMCTA: simple mean accuracy các clients (không trọng số)
            accs = [eval_res.metrics.get("accuracy", 0.0) for _, eval_res in results]
            bmcta = sum(accs) / len(accs) if accs else 0.0
            history_bmcta.append(bmcta)

            cur_c_loss = history_client_loss[-1] if history_client_loss else 0.0
            cur_s_loss = history_server_loss[-1] if history_server_loss else 0.0

            pre = metrics_agg.get("precision", 0.0)
            rec = metrics_agg.get("recall",    0.0)
            f1  = metrics_agg.get("f1_score",  0.0)
            print(f"  [Round {server_round:3d}/{num_rounds}] "
                  f"BTA={bta:.4f} BMCTA={bmcta:.4f} "
                  f"TrainLoss={cur_c_loss:.4f} TestLoss={cur_s_loss:.4f} "
                  f"Pre={pre:.4f} Rec={rec:.4f} F1={f1:.4f} ({t:.1f}s)", flush=True)

            if early_stopper and not early_stopper.update(server_round, bta):
                raise EarlyStopException()

        return agg

    strategy.aggregate_evaluate = agg_with_tracking

    def fit_config_fn(server_round: int):
        cfg = {"lr": FS["learning_rate"]}
        if strategy_name in ["fedprox", "fedprox_la"]:
            cfg["proximal_mu"] = proximal_mu_val
        if strategy_name in ["scaffold_la", "fedavg_la", "fedprox_la", "fedbn_scaffold_la", "fedbn_scaffold"]:
            cfg["tau_la"] = tau_la_val
        return cfg

    strategy.on_fit_config_fn = fit_config_fn

    t0 = time.time()
    try:
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=FS["num_clients"],
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources=client_res,
        )
    except Exception as e:
        if not isinstance(e.__cause__, EarlyStopException):
            raise

    total_time = round(time.time() - t0, 1)
    num_params = sum(p.numel() for p in net.parameters())
    model_size_mb = round(num_params * 4 / (1024 * 1024), 3)
    # Tổng băng thông truyền thông FL (upload + download) cho tất cả clients qua các rounds
    comm_per_round_mb = (num_params * 4 * 2 * FS["num_clients"]) / (1024 * 1024)
    total_comm_mb = round(comm_per_round_mb * actual_rounds[0], 1)
    total_time_min = round(total_time / 60.0, 1)

    result = {
        "dataset":           dataset_name,
        "partition_method":  partition_method,
        "strategy":          strategy_name,
        "model":             f"{model_type} ({'~570k' if model_type == '570k' else '~20k'})",
        "run":               run_id,
        "num_params":        num_params,
        "model_size_mb":     model_size_mb,
        "total_comm_mb":     total_comm_mb,
        "total_time_min":    total_time_min,
        "total_time_s":      total_time,
        "num_clients":       FS["num_clients"],
        "fl_rounds_target":  num_rounds,
        "fl_rounds_actual":  actual_rounds[0],
        "optimizer":         FS["optimizer"],
        "learning_rate":     FS["learning_rate"],
        "batch_size":        FS["batch_size"],
        "proximal_mu":       proximal_mu_val if strategy_name in ["fedprox", "fedprox_la"] else None,
        "tau_la":            tau_la_val if strategy_name in ["scaffold_la", "fedavg_la", "fedprox_la", "fedbn_scaffold_la", "fedbn_scaffold"] else None,
        # Metrics chính (theo paper)
        "BTA":               round(max(history_bta),   4) if history_bta   else 0.0,
        "bta_round":         int(np.argmax(history_bta)) + 1 if history_bta else 0,
        "BMCTA":             round(max(history_bmcta), 4) if history_bmcta else 0.0,
        "bmcta_round":       int(np.argmax(history_bmcta)) + 1 if history_bmcta else 0,
        # Loss Metrics (Best tại đỉnh BTA & Final)
        "Client_Loss":       round(history_client_loss[int(np.argmax(history_bta))], 4) if history_bta and history_client_loss and int(np.argmax(history_bta)) < len(history_client_loss) else (round(history_client_loss[-1], 4) if history_client_loss else 0.0),
        "Server_Loss":       round(history_server_loss[int(np.argmax(history_bta))], 4) if history_bta and history_server_loss and int(np.argmax(history_bta)) < len(history_server_loss) else (round(history_server_loss[-1], 4) if history_server_loss else 0.0),
        "best_client_loss":  round(min(history_client_loss), 4) if history_client_loss else 0.0,
        "best_server_loss":  round(min(history_server_loss), 4) if history_server_loss else 0.0,
        "final_client_loss": round(history_client_loss[-1], 4) if history_client_loss else 0.0,
        "final_server_loss": round(history_server_loss[-1], 4) if history_server_loss else 0.0,
        # Lịch sử từng round
        "bta_per_round":         [round(x, 4) for x in history_bta],
        "bmcta_per_round":       [round(x, 4) for x in history_bmcta],
        "client_loss_per_round": [round(x, 4) for x in history_client_loss],
        "server_loss_per_round": [round(x, 4) for x in history_server_loss],
        "round_times_s":         [round(x, 1) for x in round_times],
    }

    # Lưu kết quả ra file JSON
    run_name = f"{dataset_name}_{partition_method}_{strategy_name}_model{model_type}_run{run_id}"
    out_dir  = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Tự động vẽ đồ thị hội tụ và độ biến động (Loss, Accuracy, Volatility)
    try:
        plot_run_curves(out_dir, result)
    except Exception as e:
        print(f"  [WARNING] Không thể vẽ biểu đồ: {e}")

    # Cập nhật CSV tổng hợp ngay lập tức sau mỗi lần chạy
    update_comparison_csv(result)

    print(f"\n  [COMPLETED] -> BTA = {result['BTA']:.4f} (Round {result['bta_round']}) | BMCTA = {result['BMCTA']:.4f}")
    print(f"                 Client_Loss = {result['Client_Loss']:.4f} | Server_Loss = {result['Server_Loss']:.4f}")
    print(f"  [SAVED]     -> {out_dir / 'results.json'}")
    return result


def plot_run_curves(out_dir: Path, result: dict):
    """Vẽ và lưu đồ thị Loss Convergence, Accuracy Trajectory và Volatility (Độ biến động)."""
    import os
    os.environ["MPLCONFIGDIR"] = str(Path("results") / ".matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bta_hist = result.get("bta_per_round", [])
    bmcta_hist = result.get("bmcta_per_round", [])
    c_loss_hist = result.get("client_loss_per_round", [])
    s_loss_hist = result.get("server_loss_per_round", [])

    if not bta_hist:
        return

    rounds = list(range(1, len(bta_hist) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    strat_raw = result.get('strategy', '')
    tau_la_val = result.get('tau_la', None)
    if 'fedbn' in strat_raw and (tau_la_val == 0.0 or tau_la_val is None):
        strat_title = "FEDBN + SCAFFOLD"
    elif 'scaffold_la' in strat_raw and (tau_la_val == 0.0 or tau_la_val is None):
        strat_title = "SCAFFOLD"
    else:
        strat_title = strat_raw.upper()
    ds_title = result.get('dataset', '').upper()
    part_title = result.get('partition_method', '')
    run_idx = result.get('run', 1)
    fig.suptitle(
        f"FL Simulation: {strat_title} on {ds_title} ({part_title}, Run {run_idx})",
        fontsize=13, fontweight="bold", y=1.02
    )

    # 1. Subplot: Loss Convergence
    ax0 = axes[0]
    if c_loss_hist:
        r_c = list(range(1, len(c_loss_hist) + 1))
        ax0.plot(r_c, c_loss_hist, label="Client Train Loss", color="#1f77b4", linewidth=2)
    if s_loss_hist:
        r_s = list(range(1, len(s_loss_hist) + 1))
        ax0.plot(r_s, s_loss_hist, label="Server Test Loss", color="#d62728", linewidth=2)
        min_s_idx = int(np.argmin(s_loss_hist))
        ax0.scatter([min_s_idx + 1], [s_loss_hist[min_s_idx]], color="#d62728", s=70, zorder=5)
        ax0.annotate(f"Min: {s_loss_hist[min_s_idx]:.4f}", 
                     (min_s_idx + 1, s_loss_hist[min_s_idx]),
                     textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")
    ax0.set_title("Training & Test Loss Convergence", fontweight="bold")
    ax0.set_xlabel("FL Communication Round")
    ax0.set_ylabel("Cross-Entropy Loss")
    ax0.grid(True, linestyle="--", alpha=0.5)
    ax0.legend(loc="upper right")

    # 2. Subplot: Accuracy Trajectory
    ax1 = axes[1]
    ax1.plot(rounds, bta_hist, label="BTA (Combined / Global)", color="#2ca02c", linewidth=2)
    if bmcta_hist:
        ax1.plot(rounds, bmcta_hist, label="BMCTA (Mean Client)", color="#ff7f0e", linewidth=2, linestyle="--")
    best_bta_idx = int(np.argmax(bta_hist))
    ax1.scatter([best_bta_idx + 1], [bta_hist[best_bta_idx]], color="#2ca02c", s=100, marker="*", zorder=5)
    ax1.annotate(f"BTA: {bta_hist[best_bta_idx]:.4f} (R{best_bta_idx+1})", 
                 (best_bta_idx + 1, bta_hist[best_bta_idx]),
                 textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")
    ax1.set_title("Accuracy Trajectory (BTA & BMCTA)", fontweight="bold")
    ax1.set_xlabel("FL Communication Round")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim([0.0, 1.02])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # 3. Subplot: Volatility & Stability (Delta Loss & Rolling Std)
    ax2 = axes[2]
    target_loss = s_loss_hist if s_loss_hist else c_loss_hist
    if len(target_loss) > 1:
        deltas = np.abs(np.diff(target_loss))
        r_delta = list(range(2, len(target_loss) + 1))
        ax2.plot(r_delta, deltas, label=r"$|\Delta \mathrm{Loss}| = |L_t - L_{t-1}|$", color="#9467bd", linewidth=1.5, alpha=0.7)
        if len(target_loss) >= 5:
            rolling_std = [float(np.std(target_loss[max(0, i-4):i+1])) for i in range(len(target_loss))]
            ax2.plot(rounds, rolling_std, label="5-Round Rolling Std (Volatility)", color="#8c564b", linewidth=2)
    ax2.set_title("Round-to-Round Volatility & Stability", fontweight="bold")
    ax2.set_xlabel("FL Communication Round")
    ax2.set_ylabel(r"Loss Volatility / $\Delta$")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plot_path = out_dir / "convergence_and_volatility.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT SAVED] -> {plot_path}", flush=True)


def get_csv_key(row):
    ds = str(row.get("dataset", "")).strip()
    pt = str(row.get("partition_method", "")).strip()
    st = str(row.get("strategy", "")).strip()
    if st == "fedbn_scaffold":
        st = "fedbn_scaffold_la"
    md_str = str(row.get("model", "")).lower()
    md = "570k" if "570k" in md_str else "20k"
    run_val = str(row.get("run", "1")).strip()
    try:
        run_id = int(run_val)
    except (ValueError, TypeError):
        run_id = run_val
    return (ds, pt, st, md, run_id)


def update_comparison_csv(new_results=None):
    """Cập nhật kết quả vào comparison_table_fedsld.csv ngay sau mỗi scenario.
    Mỗi phương pháp gồm 3 lần chạy, thêm dòng tính trung bình ± độ lệch chuẩn (Mean ± Std),
    và có dòng trống cách dòng giữa các phương pháp.
    """
    if new_results is None:
        new_results = []
    elif not isinstance(new_results, list):
        new_results = [new_results]

    keys = [
        "dataset", "partition_method", "strategy", "model", "run",
        "num_params", "model_size_mb", "total_comm_mb",
        "total_time_min", "total_time_s",
        "BTA", "BMCTA", "Client_Loss", "Server_Loss", "fl_rounds_actual", "optimizer", "learning_rate"
    ]
    csv_path = RESULTS_DIR / "comparison_table_fedsld.csv"

    rows = []
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r.get("dataset")]

    row_map = {get_csv_key(r): r for r in rows}

    # Bổ sung BMCTA, Client_Loss, Server_Loss từ các file results.json cũ nếu bảng CSV chưa có
    for json_file in RESULTS_DIR.glob("*/results.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            k = get_csv_key(d)
            if k in row_map:
                if (row_map[k].get("BMCTA") in [None, "", "-"]) and "BMCTA" in d:
                    row_map[k]["BMCTA"] = str(d["BMCTA"])
                if (row_map[k].get("Client_Loss") in [None, "", "-"]) and "Client_Loss" in d:
                    row_map[k]["Client_Loss"] = str(d["Client_Loss"])
                if (row_map[k].get("Server_Loss") in [None, "", "-"]) and "Server_Loss" in d:
                    row_map[k]["Server_Loss"] = str(d["Server_Loss"])
            else:
                row_map[k] = {col: str(d.get(col, "-")) for col in keys}
        except Exception:
            pass

    for nr in new_results:
        k = get_csv_key(nr)
        row_map[k] = {col: str(nr.get(col, "-")) for col in keys}

    target_strategies = [
        "fedavg", "fedavg_la", "fedprox", "fedprox_la",
        "fednova", "scaffold", "scaffold_la", "fednova_scaffold",
        "fedbn_scaffold_la"
    ]
    final_rows = []
    first_group = True

    for ds in ["organmnist_axial", "pathmnist"]:
        for st in target_strategies:
            if st == "fedbn_scaffold_la":
                num_params = 570299 if ds == "organmnist_axial" else 570897
            else:
                num_params = 570107 if ds == "organmnist_axial" else 570705
            model_size = 2.175 if ds == "organmnist_axial" else 2.177

            # Dòng trống cách dòng giữa các phương pháp
            if not first_group:
                final_rows.append({col: "" for col in keys})
            first_group = False

            bta_vals = []
            bmcta_vals = []
            closs_vals = []
            sloss_vals = []
            time_vals = []
            comm_val = "-"
            rounds_val = "-"

            # Ghi nhận 3 runs
            for r in [1, 2, 3]:
                k = (ds, "practical_noniid", st, "570k", r)
                if k in row_map:
                    row_data = {col: (row_map[k].get(col) if row_map[k].get(col) not in [None, ""] else "-") for col in keys}
                    final_rows.append(row_data)
                    bta_str = row_data.get("BTA", "-")
                    if bta_str not in ["-", "", None]:
                        try:
                            bta_vals.append(float(bta_str))
                        except ValueError:
                            pass
                    bmcta_str = row_data.get("BMCTA", "-")
                    if bmcta_str not in ["-", "", None]:
                        try:
                            bmcta_vals.append(float(bmcta_str))
                        except ValueError:
                            pass
                    closs_str = row_data.get("Client_Loss", "-")
                    if closs_str not in ["-", "", None]:
                        try:
                            closs_vals.append(float(closs_str))
                        except ValueError:
                            pass
                    sloss_str = row_data.get("Server_Loss", "-")
                    if sloss_str not in ["-", "", None]:
                        try:
                            sloss_vals.append(float(sloss_str))
                        except ValueError:
                            pass
                    time_str = row_data.get("total_time_min", "-")
                    if time_str not in ["-", "", None]:
                        try:
                            time_vals.append(float(time_str))
                        except ValueError:
                            pass
                    if row_data.get("total_comm_mb", "-") != "-":
                        comm_val = row_data["total_comm_mb"]
                    if row_data.get("fl_rounds_actual", "-") != "-":
                        rounds_val = row_data["fl_rounds_actual"]
                else:
                    final_rows.append({
                        "dataset": ds,
                        "partition_method": "practical_noniid",
                        "strategy": st,
                        "model": "570k (~570k)",
                        "run": str(r),
                        "num_params": num_params,
                        "model_size_mb": model_size,
                        "total_comm_mb": "-",
                        "total_time_min": "-",
                        "total_time_s": "-",
                        "BTA": "-",
                        "BMCTA": "-",
                        "Client_Loss": "-",
                        "Server_Loss": "-",
                        "fl_rounds_actual": "-",
                        "optimizer": "sgd",
                        "learning_rate": "0.01",
                    })

            # Tính dòng Mean ± Std
            if len(bta_vals) == 3:
                mean_bta = float(np.mean(bta_vals))
                std_bta = float(np.std(bta_vals, ddof=1))
                summary_bta = f"{mean_bta:.4f} ± {std_bta:.4f}"
            elif len(bta_vals) > 0:
                mean_bta = float(np.mean(bta_vals))
                summary_bta = f"{mean_bta:.4f} ({len(bta_vals)}/3 runs)"
            else:
                summary_bta = "-"

            if len(bmcta_vals) == 3:
                mean_bmcta = float(np.mean(bmcta_vals))
                std_bmcta = float(np.std(bmcta_vals, ddof=1))
                summary_bmcta = f"{mean_bmcta:.4f} ± {std_bmcta:.4f}"
            elif len(bmcta_vals) > 0:
                mean_bmcta = float(np.mean(bmcta_vals))
                summary_bmcta = f"{mean_bmcta:.4f} ({len(bmcta_vals)}/3 runs)"
            else:
                summary_bmcta = "-"

            if len(closs_vals) == 3:
                mean_closs = float(np.mean(closs_vals))
                std_closs = float(np.std(closs_vals, ddof=1))
                summary_closs = f"{mean_closs:.4f} ± {std_closs:.4f}"
            elif len(closs_vals) > 0:
                mean_closs = float(np.mean(closs_vals))
                summary_closs = f"{mean_closs:.4f} ({len(closs_vals)}/3 runs)"
            else:
                summary_closs = "-"

            if len(sloss_vals) == 3:
                mean_sloss = float(np.mean(sloss_vals))
                std_sloss = float(np.std(sloss_vals, ddof=1))
                summary_sloss = f"{mean_sloss:.4f} ± {std_sloss:.4f}"
            elif len(sloss_vals) > 0:
                mean_sloss = float(np.mean(sloss_vals))
                summary_sloss = f"{mean_sloss:.4f} ({len(sloss_vals)}/3 runs)"
            else:
                summary_sloss = "-"

            if time_vals:
                avg_time_min = float(np.mean(time_vals))
                avg_time_min_str = f"{avg_time_min:.1f}"
                avg_time_s_str = f"{avg_time_min * 60:.1f}"
            else:
                avg_time_min_str = "-"
                avg_time_s_str = "-"

            summary_row = {
                "dataset": ds,
                "partition_method": "practical_noniid",
                "strategy": st,
                "model": "570k (~570k)",
                "run": "Mean ± Std",
                "num_params": num_params,
                "model_size_mb": model_size,
                "total_comm_mb": comm_val,
                "total_time_min": avg_time_min_str,
                "total_time_s": avg_time_s_str,
                "BTA": summary_bta,
                "BMCTA": summary_bmcta,
                "Client_Loss": summary_closs,
                "Server_Loss": summary_sloss,
                "fl_rounds_actual": rounds_val,
                "optimizer": "sgd",
                "learning_rate": "0.01",
            }
            final_rows.append(summary_row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(final_rows)
    print(f"\n  [CSV UPDATED] -> {csv_path}", flush=True)


def main():
    valid_strategies = list(set(
        FS.get("fl_strategies", ["fedavg", "fedprox"]) +
        ["fedbn", "fednova", "scaffold", "scaffold_la", "fednova_scaffold", "fedavg_la", "fedprox_la", "fedbn_scaffold_la"]
    ))

    parser = argparse.ArgumentParser(description="FedSLD Reproduction Simulation")
    parser.add_argument("--dataset",        choices=list(DATASET_FNS.keys()), default="organmnist_axial")
    parser.add_argument("--partition",      choices=["practical_noniid", "pathological_noniid"], default="practical_noniid")
    parser.add_argument("--strategy",       choices=valid_strategies, default="fedavg")
    parser.add_argument("--model",          choices=["20k", "570k"], default="20k", help="Model architecture: 20k (ours) or 570k (FedSLD paper)")
    parser.add_argument("--run_id",         type=int, default=1, choices=[1, 2, 3], help="Run iteration (1, 2, or 3, default: 1)")
    parser.add_argument("--runs",           nargs="+", type=int, default=None, help="List of run IDs to execute, e.g. --runs 2 3")
    parser.add_argument("--all_runs",       action="store_true", help="Run all 3 runs (run 1, 2, 3) sequentially")
    parser.add_argument("--num_rounds",     type=int, default=None, help="Override number of FL rounds (useful for smoke tests)")
    parser.add_argument("--proximal_mu",    type=float, default=None, help="Proximal mu for FedProx (default from config: 0.1)")
    parser.add_argument("--tau_la",         type=float, default=None, help="Tau parameter for Logit Adjustment in SCAFFOLD-LA (default from config: 1.0)")
    parser.add_argument("--all",            action="store_true", help="Run all experiment scenarios")
    parser.add_argument("--practical_570k", action="store_true", help="Run all 4 practical non-IID scenarios for 570K model (OrganMNIST + PathMNIST x FedAvg + FedProx)")
    parser.add_argument("--all_570k",       action="store_true", help="Run all 8 scenarios for 570K model (Practical 80 rds + Pathological 160 rds)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.runs:
        runs_to_exec = args.runs
    elif args.all_runs:
        runs_to_exec = [1, 2, 3]
    else:
        runs_to_exec = [args.run_id]

    if args.all_570k:
        print("\n" + "="*70)
        print("  BATCH RUN: ALL 8 SCENARIOS (570K PARAMS)")
        print("  Part 1: Practical Non-IID (80 rounds each)")
        print("    1. OrganMNIST-Axial + Practical Non-IID + FedAvg  + 570K")
        print("    2. OrganMNIST-Axial + Practical Non-IID + FedProx + 570K")
        print("    3. PathMNIST        + Practical Non-IID + FedAvg  + 570K")
        print("    4. PathMNIST        + Practical Non-IID + FedProx + 570K")
        print("  Part 2: Pathological Non-IID (160 rounds each)")
        print("    5. OrganMNIST-Axial + Pathological Non-IID + FedAvg  + 570K")
        print("    6. OrganMNIST-Axial + Pathological Non-IID + FedProx + 570K")
        print("    7. PathMNIST        + Pathological Non-IID + FedAvg  + 570K")
        print("    8. PathMNIST        + Pathological Non-IID + FedProx + 570K")
        print("="*70 + "\n")
        scenarios = [
            ("organmnist_axial", "practical_noniid",    "fedavg",  "570k"),
            ("organmnist_axial", "practical_noniid",    "fedprox", "570k"),
            ("pathmnist",        "practical_noniid",    "fedavg",  "570k"),
            ("pathmnist",        "practical_noniid",    "fedprox", "570k"),
            ("organmnist_axial", "pathological_noniid", "fedavg",  "570k"),
            ("organmnist_axial", "pathological_noniid", "fedprox", "570k"),
            ("pathmnist",        "pathological_noniid", "fedavg",  "570k"),
            ("pathmnist",        "pathological_noniid", "fedprox", "570k"),
        ]
        for idx, (ds, pt, st, md) in enumerate(scenarios, 1):
            for r in runs_to_exec:
                print(f"\n>>> [SCENARIO {idx}/8 | RUN {r}/3] STARTING: {ds.upper()} | {pt} | {st.upper()} | {md.upper()} ...", flush=True)
                run_one(ds, pt, st, device, model_type=md, run_id=r, custom_num_rounds=args.num_rounds, custom_proximal_mu=args.proximal_mu, custom_tau_la=args.tau_la)
                print(f">>> [SCENARIO {idx}/8 | RUN {r}/3] FINISHED: {ds.upper()} | {pt} | {st.upper()} | {md.upper()}\n", flush=True)

    elif args.practical_570k:
        print("\n" + "="*70)
        print("  BATCH RUN: 4 PRACTICAL NON-IID SCENARIOS (570K PARAMS, 80 ROUNDS)")
        print("  1. OrganMNIST-Axial + Practical Non-IID + FedAvg  + 570K (80 rounds)")
        print("  2. OrganMNIST-Axial + Practical Non-IID + FedProx + 570K (80 rounds)")
        print("  3. PathMNIST        + Practical Non-IID + FedAvg  + 570K (80 rounds)")
        print("  4. PathMNIST        + Practical Non-IID + FedProx + 570K (80 rounds)")
        print("="*70 + "\n")
        scenarios = [
            ("organmnist_axial", "practical_noniid", "fedavg",  "570k"),
            ("organmnist_axial", "practical_noniid", "fedprox", "570k"),
            ("pathmnist",        "practical_noniid", "fedavg",  "570k"),
            ("pathmnist",        "practical_noniid", "fedprox", "570k"),
        ]
        for idx, (ds, pt, st, md) in enumerate(scenarios, 1):
            for r in runs_to_exec:
                print(f"\n>>> [SCENARIO {idx}/4 | RUN {r}/3] STARTING: {ds.upper()} | {pt} | {st.upper()} | {md.upper()} ...", flush=True)
                run_one(ds, pt, st, device, model_type=md, run_id=r, custom_num_rounds=args.num_rounds, custom_proximal_mu=args.proximal_mu, custom_tau_la=args.tau_la)
                print(f">>> [SCENARIO {idx}/4 | RUN {r}/3] FINISHED: {ds.upper()} | {pt} | {st.upper()} | {md.upper()}\n", flush=True)

    elif args.all:
        for ds in list(DATASET_FNS.keys()):
            for pt in ["practical_noniid", "pathological_noniid"]:
                for st in FS.get("fl_strategies", ["fedavg", "fedprox"]):
                    for r in runs_to_exec:
                        run_one(ds, pt, st, device, model_type=args.model, run_id=r, custom_num_rounds=args.num_rounds, custom_proximal_mu=args.proximal_mu, custom_tau_la=args.tau_la)
    else:
        for r in runs_to_exec:
            run_one(args.dataset, args.partition, args.strategy, device, model_type=args.model, run_id=r, custom_num_rounds=args.num_rounds, custom_proximal_mu=args.proximal_mu, custom_tau_la=args.tau_la)


if __name__ == "__main__":
    main()
