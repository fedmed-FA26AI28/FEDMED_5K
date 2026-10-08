"""
FL Server thực - Chạy trên PC, chờ Jetson Client kết nối qua mạng Wi-Fi.

Cách dùng:
    python -m experiments.run_server --strategy fedavg --num_clients 1 --rounds 20
"""

import argparse
import json
import math
import datetime
import hashlib
import time
import sys
from pathlib import Path

import flwr as fl
import torch
import yaml
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from models.cnn import build_model, get_parameters, set_parameters
from server.server import get_strategy, EarlyStoppingCallback, GlobalLRScheduler

# ─── Đọc config ──────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "experiment.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

LR         = CFG.get("learning_rate", 0.001)
NUM_ROUNDS = CFG.get("fl_rounds", 20)
ES_CFG     = CFG.get("early_stopping", {})
LR_CFG     = CFG.get("lr_scheduler", {})

# Thư mục lưu kết quả riêng cho Jetson (tách biệt với simulate_federated)
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "real_federated"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description="FedMedAI - FL Server (Real Deployment)")
    parser.add_argument("--strategy",    type=str, default="fedavg",
                        choices=["fedavg", "fedprox", "coverage", "balanced", "logit_only", "head_only", "vacant_distill", "coverage_distill"])
    parser.add_argument("--model", choices=["legacy", "tiny_cnn", "tiny_cnn_gn", "mobilenet_v3_small"], default="legacy")
    parser.add_argument("--size", type=int, choices=[28, 64], default=28)
    parser.add_argument("--logit_tau", type=float, default=1.0)
    parser.add_argument("--head_mu", type=float, default=0.01)
    parser.add_argument("--coverage_kappa", type=float, default=32.0)
    parser.add_argument("--distill_mu", type=float, default=0.1)
    parser.add_argument("--distill_temperature", type=float, default=2.0)
    parser.add_argument("--distill_max_count", type=int, default=0)
    parser.add_argument("--distill_warmup_rounds", type=int, default=1)
    parser.add_argument("--num_clients", type=int, default=1,
                        help="Số lượng client Jetson sẽ kết nối")
    parser.add_argument("--rounds",      type=int, default=NUM_ROUNDS,
                        help="Số round FL")
    parser.add_argument("--alpha",       type=float, default=1.0,
                        help="Mức Non-IID alpha của partition (để ghi vào kết quả)")
    parser.add_argument("--host",        type=str, default="0.0.0.0",
                        help="IP lắng nghe (mặc định 0.0.0.0 = tất cả interface)")
    parser.add_argument("--port",        type=int, default=8080,
                        help="Port gRPC (mặc định 8080)")
    args = parser.parse_args()
    if args.strategy in {"vacant_distill", "coverage_distill"}:
        if not math.isfinite(args.distill_mu) or args.distill_mu <= 0:
            parser.error("--distill_mu must be finite and positive")
        if not math.isfinite(args.distill_temperature) or args.distill_temperature <= 0:
            parser.error("--distill_temperature must be finite and positive")
        if args.distill_max_count < 0 or args.distill_warmup_rounds < 0:
            parser.error("distillation count and warmup must be non-negative")
        if args.rounds <= args.distill_warmup_rounds:
            parser.error("--rounds must exceed --distill_warmup_rounds")

    # ─── Khởi tạo Global Model ───────────────────────────────────────────────
    net         = build_model(args.model, num_classes=8)
    init_params = ndarrays_to_parameters(get_parameters(net))

    # ─── Khởi tạo Strategy ───────────────────────────────────────────────────
    strategy = get_strategy(
        strategy_name=args.strategy,
        num_clients=args.num_clients,
        initial_parameters=init_params,
    )

    # ─── Theo dõi per-round ──────────────────────────────────────────────────
    early_stopper = EarlyStoppingCallback(
        patience=ES_CFG.get("patience", 5),
        min_delta=ES_CFG.get("min_delta", 0.001)
    )
    lr_scheduler = GlobalLRScheduler(
        initial_lr=LR,
        factor=LR_CFG.get("factor", 0.5),
        patience=LR_CFG.get("patience", 3),
        min_lr=LR_CFG.get("min_lr", 1e-6)
    )

    history_acc = []
    round_times = []
    orig_agg    = strategy.aggregate_evaluate

    def agg_with_tracking(server_round, results, failures):
        t_start = time.time()
        agg     = orig_agg(server_round, results, failures)
        t_round = time.time() - t_start

        if agg is not None:
            _, metrics_agg = agg
            acc = metrics_agg.get("accuracy",  0.0)
            pre = metrics_agg.get("precision", 0.0)
            rec = metrics_agg.get("recall",    0.0)
            f1  = metrics_agg.get("f1_score",  0.0)

            history_acc.append(acc)
            round_times.append(t_round)

            new_lr = lr_scheduler.step(acc)
            print(f"  [Round {server_round:3d}] "
                  f"Acc={acc:.4f} Pre={pre:.4f} "
                  f"Rec={rec:.4f} F1={f1:.4f} "
                  f"LR={new_lr:.6f} t={t_round:.1f}s")

            if not early_stopper.update(server_round, acc):
                print(f"\n  [Early Stop] Dung tai Round {server_round}. "
                      f"Best Acc={early_stopper.best_acc:.4f}")

        return agg

    strategy.aggregate_evaluate = agg_with_tracking
    def fit_config(server_round):
        config = lr_scheduler.get_config()
        config.update({"server_round": server_round,
                       "balanced_sampling": args.strategy == "balanced"})
        if args.strategy in {"coverage", "logit_only", "head_only", "coverage_distill"}:
            config.update({
                "logit_tau": args.logit_tau if args.strategy in {"coverage", "logit_only", "coverage_distill"} else 0.0,
                "head_mu": args.head_mu if args.strategy in {"coverage", "head_only", "coverage_distill"} else 0.0,
                "coverage_kappa": args.coverage_kappa,
            })
        if args.strategy in {"vacant_distill", "coverage_distill"}:
            config.update({
                "distill_mu": args.distill_mu,
                "distill_temperature": args.distill_temperature,
                "distill_max_count": args.distill_max_count,
                "distill_warmup_rounds": args.distill_warmup_rounds,
            })
        return config
    strategy.on_fit_config_fn = fit_config
    strategy.latest_parameters = init_params
    original_aggregate_fit = strategy.aggregate_fit
    def aggregate_fit(server_round, results, failures):
        aggregated = original_aggregate_fit(server_round, results, failures)
        if aggregated is not None and aggregated[0] is not None:
            strategy.latest_parameters = aggregated[0]
        return aggregated
    strategy.aggregate_fit = aggregate_fit

    # ─── Khởi động Server ────────────────────────────────────────────────────
    server_address = f"{args.host}:{args.port}"
    print("=" * 60)
    print(f"  FL SERVER DANG KHOI DONG")
    print(f"  Strategy   : {args.strategy.upper()}")
    print(f"  Clients    : {args.num_clients}")
    print(f"  Rounds     : {args.rounds}")
    print(f"  Listening  : {server_address}")
    print(f"  >>> Dang cho Jetson Client ket noi... <<<")
    print("=" * 60)

    t0 = time.time()
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    total_time = time.time() - t0

    # ─── Lưu kết quả vào results/real_federated/ ─────────────────────────────
    best_acc        = float(max(history_acc)) if history_acc else 0.0
    convergence_rnd = (history_acc.index(best_acc) + 1) if history_acc else 0
    final_acc       = float(history_acc[-1]) if history_acc else 0.0

    result = {
        "strategy":             args.strategy,
        "num_clients":          args.num_clients,
        "alpha":                args.alpha,
        "fl_rounds_target":     args.rounds,
        "fl_rounds_actual":     len(history_acc),
        "best_accuracy":        best_acc,
        "final_accuracy":       final_acc,
        "convergence_round":    convergence_rnd,
        "total_time_s":         round(total_time, 1),
        "avg_time_per_round_s": round(sum(round_times) / len(round_times), 1) if round_times else 0,
        "deployment":           "real_jetson",
        "history_accuracy":     history_acc,
    }

    # Tên thư mục: real_clients1_alpha1.0_fedavg
    run_name = (f"real_clients{args.num_clients}_alpha{args.alpha}_{args.strategy}_"
                f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    out_dir  = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    final_arrays = parameters_to_ndarrays(strategy.latest_parameters)
    set_parameters(net, final_arrays)
    torch.save(net.state_dict(), out_dir / "global_model.pt")
    result["model"] = args.model
    result["size"] = args.size
    result["global_parameter_sha256"] = hashlib.sha256(
        b"".join(array.tobytes() for array in final_arrays)
    ).hexdigest()
    result["final_test_metrics"] = None  # Physical integration is validation-only.

    out_file = out_dir / "results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n  Ket qua da luu vao: {out_file}")
    print(f"  Best Acc = {best_acc:.4f} tai Round {convergence_rnd}")
    print(f"  Tong thoi gian: {total_time:.1f}s")


if __name__ == "__main__":
    main()
