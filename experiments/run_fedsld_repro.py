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
import time
from pathlib import Path
import numpy as np
import yaml
import torch
import flwr as fl

from datasets.medmnist_code import get_pathmnist_datasets, get_organmnist_datasets
from datasets.partition import load_partition
from models.cnn import SimpleCNN, get_parameters
from models.cnn_570k import CNN570K
from client.client import make_client_fn, make_fedbn_client_fn
from client.client_fednova import make_fednova_client_fn
from client.client_scaffold import make_scaffold_client_fn
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


def run_one(dataset_name, partition_method, strategy_name, device, model_type="20k", run_id=1, custom_num_rounds=None, custom_proximal_mu=None):
    get_fn, num_classes = DATASET_FNS[dataset_name]
    num_rounds = custom_num_rounds if custom_num_rounds is not None else FS["partition_settings"][partition_method]["fl_rounds"]
    proximal_mu_val = custom_proximal_mu if custom_proximal_mu is not None else float(FS.get("proximal_mu", 0.1))

    model_label = "~570K params (FedSLD Paper)" if model_type == "570k" else "~20K params (Ours)"
    print(f"\n{'='*65}")
    print(f"  DATASET  : {dataset_name.upper()}")
    print(f"  PARTITION: {partition_method}")
    if strategy_name == "fedprox":
        print(f"  STRATEGY : {strategy_name.upper()} (proximal_mu={proximal_mu_val})")
    else:
        print(f"  STRATEGY : {strategy_name.upper()}")
    print(f"  MODEL    : {model_type.upper()} ({model_label})")
    print(f"  RUN      : {run_id}/3 (Seed 42)")
    print(f"  CLIENTS  : {FS['num_clients']} | BATCH SIZE: {FS['batch_size']}")
    print(f"  ROUNDS   : {num_rounds} | LR: {FS['learning_rate']} ({FS['optimizer'].upper()})")
    print(f"{'='*65}")

    # Dataset
    train_ds, val_ds, test_ds, _ = get_fn(download=False)

    # Partition
    json_file = PARTITION_DIR / f"{partition_method}_{dataset_name}_seed42_clients12.json"
    if not json_file.exists():
        print(f"  [SKIP] Partition file không tồn tại: {json_file}")
        print(f"  -> Chạy trước: python -m datasets.partition_fedsld")
        return None
    partition = load_partition(str(json_file))

    # Model
    in_channels = train_ds[0][0].shape[0]
    if model_type == "570k":
        net = CNN570K(in_channels=in_channels, num_classes=num_classes).to(device)
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
    elif strategy_name == "fednova_scaffold":
        client_fn_maker = make_fednova_scaffold_client_fn
    else:
        client_fn_maker = make_client_fn

    client_fn = client_fn_maker(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        partition=partition,
        local_epochs=FS["local_epochs"],
        lr=FS["learning_rate"],
        device=device,
        num_classes=num_classes,
        batch_size=FS["batch_size"],
        optimizer_type=FS["optimizer"],
        momentum=FS.get("momentum", 0.0),
        weight_decay=FS.get("weight_decay", 0.0),
        model_type=model_type,
    )

    # Tracking metrics
    history_bta   = []   # Best Test Accuracy per round (weighted avg)
    history_bmcta = []   # Best Mean Client Test Accuracy per round (simple mean)
    round_times   = []
    actual_rounds = [0]
    round_start   = [time.time()]
    es_cfg        = FS.get("early_stopping", {"enabled": False})

    early_stopper = EarlyStoppingCallback(
        patience=es_cfg.get("patience", 10),
        min_delta=es_cfg.get("min_delta", 0.001)
    ) if es_cfg.get("enabled", False) else None

    orig_agg = strategy.aggregate_evaluate

    def agg_with_tracking(server_round, results, failures):
        t = time.time() - round_start[0]
        round_times.append(t)
        round_start[0] = time.time()
        actual_rounds[0] = server_round

        agg = orig_agg(server_round, results, failures)
        if agg is not None and results:
            _, metrics_agg = agg

            # BTA: weighted average accuracy (toàn bộ test set)
            bta = metrics_agg.get("accuracy", 0.0)
            history_bta.append(bta)

            # BMCTA: simple mean accuracy các clients (không trọng số)
            accs = [eval_res.metrics.get("accuracy", 0.0) for _, eval_res in results]
            bmcta = sum(accs) / len(accs) if accs else 0.0
            history_bmcta.append(bmcta)

            pre = metrics_agg.get("precision", 0.0)
            rec = metrics_agg.get("recall",    0.0)
            f1  = metrics_agg.get("f1_score",  0.0)
            print(f"  [Round {server_round:3d}/{num_rounds}] "
                  f"BTA={bta:.4f} "
                  f"Pre={pre:.4f} Rec={rec:.4f} F1={f1:.4f} ({t:.1f}s)", flush=True)

            if early_stopper and not early_stopper.update(server_round, bta):
                raise EarlyStopException()

        return agg

    strategy.aggregate_evaluate = agg_with_tracking

    def fit_config_fn(server_round: int):
        cfg = {"lr": FS["learning_rate"]}
        if strategy_name == "fedprox":
            cfg["proximal_mu"] = proximal_mu_val
        return cfg

    strategy.on_fit_config_fn = fit_config_fn

    t0 = time.time()
    try:
        # Tối ưu hóa tài nguyên cho từng dataset:
        # - OrganMNIST (nhẹ: 34k ảnh 1 kênh): num_gpus=0 để Ray chạy song song đa tiến trình (~20-25s/round).
        # - PathMNIST (90k ảnh 3 kênh): num_gpus=0.5 (chạy 2 client song song trên GPU 8GB, 86s/round, ổn định 100%).
        if dataset_name == "organmnist_axial":
            client_res = {"num_cpus": 2, "num_gpus": 0}
        else:
            gpu_res = 0.5 if device.type == "cuda" else 0.0
            client_res = {"num_cpus": 2, "num_gpus": gpu_res}

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
        "proximal_mu":       proximal_mu_val if strategy_name == "fedprox" else None,
        # Metrics chính (theo paper)
        "BTA":               round(max(history_bta),   4) if history_bta   else 0.0,
        "BMCTA":             round(max(history_bmcta), 4) if history_bmcta else 0.0,
        # Lịch sử từng round
        "bta_per_round":     [round(x, 4) for x in history_bta],
        "bmcta_per_round":   [round(x, 4) for x in history_bmcta],
        "round_times_s":     [round(x, 1) for x in round_times],
    }

    # Lưu kết quả ra file JSON
    run_name = f"{dataset_name}_{partition_method}_{strategy_name}_model{model_type}_run{run_id}"
    out_dir  = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Cập nhật CSV tổng hợp ngay lập tức sau mỗi lần chạy
    update_comparison_csv(result)

    print(f"\n  [COMPLETED] -> BTA = {result['BTA']:.4f}")
    print(f"  [SAVED]     -> {out_dir / 'results.json'}")
    return result


def get_csv_key(row):
    ds = str(row.get("dataset", "")).strip()
    pt = str(row.get("partition_method", "")).strip()
    st = str(row.get("strategy", "")).strip()
    md_str = str(row.get("model", "")).lower()
    md = "570k" if "570k" in md_str else "20k"
    run_val = str(row.get("run", "1")).strip()
    try:
        run_id = int(run_val)
    except (ValueError, TypeError):
        run_id = run_val
    return (ds, pt, st, md, run_id)


def update_comparison_csv(new_results):
    """Cập nhật kết quả vào comparison_table_fedsld.csv ngay sau mỗi scenario.
    Mỗi phương pháp gồm 3 lần chạy, thêm dòng tính trung bình ± độ lệch chuẩn (Mean ± Std),
    và có dòng trống cách dòng giữa các phương pháp.
    """
    if not new_results:
        return
    if not isinstance(new_results, list):
        new_results = [new_results]

    keys = [
        "dataset", "partition_method", "strategy", "model", "run",
        "num_params", "model_size_mb", "total_comm_mb",
        "total_time_min", "total_time_s",
        "BTA", "fl_rounds_actual", "optimizer", "learning_rate"
    ]
    csv_path = RESULTS_DIR / "comparison_table_fedsld.csv"

    rows = []
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r.get("dataset")]

    row_map = {get_csv_key(r): r for r in rows}

    for nr in new_results:
        k = get_csv_key(nr)
        row_map[k] = {col: str(nr.get(col, "-")) for col in keys}

    target_strategies = ["fedavg", "fedprox", "fednova", "scaffold", "fednova_scaffold"]
    final_rows = []
    first_group = True

    for ds in ["organmnist_axial", "pathmnist"]:
        num_params = 570107 if ds == "organmnist_axial" else 570705
        model_size = 2.175 if ds == "organmnist_axial" else 2.177

        for st in target_strategies:
            # Dòng trống cách dòng giữa các phương pháp
            if not first_group:
                final_rows.append({col: "" for col in keys})
            first_group = False

            bta_vals = []
            time_vals = []
            comm_val = "-"
            rounds_val = "-"

            # Ghi nhận 3 runs
            for r in [1, 2, 3]:
                k = (ds, "practical_noniid", st, "570k", r)
                if k in row_map:
                    row_data = row_map[k]
                    final_rows.append(row_data)
                    bta_str = row_data.get("BTA", "-")
                    if bta_str not in ["-", "", None]:
                        try:
                            bta_vals.append(float(bta_str))
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
        ["fedbn", "fednova", "scaffold", "fednova_scaffold"]
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
                run_one(ds, pt, st, device, model_type=md, run_id=r, custom_num_rounds=args.num_rounds, custom_proximal_mu=args.proximal_mu)
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
                run_one(ds, pt, st, device, model_type=md, run_id=r, custom_num_rounds=args.num_rounds, custom_proximal_mu=args.proximal_mu)
                print(f">>> [SCENARIO {idx}/4 | RUN {r}/3] FINISHED: {ds.upper()} | {pt} | {st.upper()} | {md.upper()}\n", flush=True)

    elif args.all:
        for ds in list(DATASET_FNS.keys()):
            for pt in ["practical_noniid", "pathological_noniid"]:
                for st in FS.get("fl_strategies", ["fedavg", "fedprox"]):
                    for r in runs_to_exec:
                        run_one(ds, pt, st, device, model_type=args.model, run_id=r, custom_num_rounds=args.num_rounds, custom_proximal_mu=args.proximal_mu)
    else:
        for r in runs_to_exec:
            run_one(args.dataset, args.partition, args.strategy, device, model_type=args.model, run_id=r, custom_num_rounds=args.num_rounds, custom_proximal_mu=args.proximal_mu)


if __name__ == "__main__":
    main()
