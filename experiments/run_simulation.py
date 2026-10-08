"""
Phase 5: Flower FL Simulation tren PC.
Toan bo cau hinh doc tu configs/experiment.yaml.

Chay:
    python -m experiments.run_simulation --all
    python -m experiments.run_simulation --num_clients 3 --alpha 1.0 --strategy fedavg
"""

import argparse
import datetime
import hashlib
import json
import time
from pathlib import Path

import yaml
import torch
import flwr as fl

from datasets.medmnist_code import (
    build_transform, fit_train_normalization, get_bloodmnist_dataset,
)
from datasets.partition import (
    dirichlet_partition,
    load_partition,
    save_partition,
    stratified_holdout_indices,
    stratified_subsample_indices,
    stratified_validation_partition,
)
from models.cnn import build_model, get_parameters, set_parameters
from client.client import make_client_fn
from client.evaluate import evaluate
from server.server import get_strategy, EarlyStoppingCallback, GlobalLRScheduler
from flwr.common import parameters_to_ndarrays
from torch.utils.data import DataLoader


def get_gpu_info() -> dict:
    """Thu thap thong tin GPU truoc khi chay."""
    if not torch.cuda.is_available():
        return {
            "gpu_available": False,
            "gpu_name": "N/A (CPU only)",
            "gpu_total_memory_gb": 0.0,
        }
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_available":        True,
        "gpu_name":             props.name,
        "gpu_total_memory_gb":  round(props.total_memory / (1024**3), 2),
        "cuda_version":         torch.version.cuda,
        "cudnn_version":        torch.backends.cudnn.version(),
    }


def get_gpu_usage() -> dict:
    """Doc muc su dung GPU tai thoi diem hien tai va peak."""
    if not torch.cuda.is_available():
        return {"peak_gpu_memory_mb": 0.0, "current_gpu_memory_mb": 0.0}
    return {
        "peak_gpu_memory_mb":    round(torch.cuda.max_memory_allocated(0) / (1024**2), 1),
        "current_gpu_memory_mb": round(torch.cuda.memory_allocated(0)     / (1024**2), 1),
    }

# === Doc cau hinh tu experiment.yaml (1 nguon su that duy nhat) ===
_CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "experiment.yaml"

with open(_CFG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

# Lay cac gia tri tu CFG
DEFAULT_NUM_CLIENTS = CFG["num_clients_options"]        # [3, 5, 10]
DEFAULT_ALPHAS      = CFG["alpha_values"]               # [1.0, 0.3, 0.1]
DEFAULT_STRATEGIES  = CFG["fl_strategies"]              # list 11 strategies
DEFAULT_SEED        = CFG["seed"]                       # 42
PARTITION_DIR       = Path(CFG["partition_save_dir"])   # data/partitions
RESULTS_DIR         = Path(CFG["results_dir"]) / "research"
LOCAL_EPOCHS        = CFG["local_epochs"]               # 5
NUM_ROUNDS          = CFG["fl_rounds"]                  # 50
LR                  = CFG["learning_rate"]              # 0.001

ES_PATIENCE         = CFG["early_stopping"]["patience"]    # 5
ES_MIN_DELTA        = CFG["early_stopping"]["min_delta"]   # 0.001

LR_FACTOR           = CFG["lr_scheduler"]["factor"]        # 0.5
LR_PATIENCE         = CFG["lr_scheduler"]["patience"]      # 3
LR_MIN              = CFG["lr_scheduler"]["min_lr"]        # 0.000001

CLIENT_RESOURCES    = CFG["client_resources"]              # {num_cpus: 2, num_gpus: 0.5}


class EarlyStopException(Exception):
    """Exception de ep Flower dung vong lap."""
    pass

def calc_model_size_bytes(model) -> int:
    return sum(arr.nbytes for arr in get_parameters(model))


def _partition_hash(indices) -> str:
    import numpy as np

    values = np.asarray(sorted(indices), dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def evaluate_final_global_model(
    parameters,
    num_classes,
    device,
    batch_size,
    calibration_indices=None,
    conformal_alpha=0.1,
    model_name="legacy",
    size=28,
    normalization=None,
):
    """Calibrate on held-out validation, then evaluate test exactly once."""
    from monitoring.reliability import (
        collect_logits,
        conformal_metrics,
        fit_conformal_threshold,
        fit_temperature,
        probability_metrics,
    )
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    from torch.utils.data import Subset

    model = build_model(model_name, num_classes=num_classes).to(device)
    set_parameters(model, parameters_to_ndarrays(parameters))
    dataset_options = {}
    if size != 28:
        dataset_options["size"] = size
    if normalization is not None:
        dataset_options["normalization"] = normalization
    temperature = 1.0
    threshold = None
    if calibration_indices is not None:
        val_dataset, _ = get_bloodmnist_dataset("val", download=True, **dataset_options)
        calibration_loader = DataLoader(
            Subset(val_dataset, calibration_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        calibration_logits, calibration_labels = collect_logits(
            model, calibration_loader, device
        )
        temperature = fit_temperature(calibration_logits, calibration_labels)
        threshold = fit_conformal_threshold(
            calibration_logits,
            calibration_labels,
            alpha=conformal_alpha,
            temperature=temperature,
        )

    test_dataset, _ = get_bloodmnist_dataset("test", download=True, **dataset_options)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_logits, test_labels = collect_logits(model, test_loader, device)
    predictions = test_logits.argmax(dim=1)
    y_true, y_pred = test_labels.tolist(), predictions.tolist()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0
    )
    _, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0
    )
    result = {
        "loss": float(torch.nn.functional.cross_entropy(test_logits, test_labels).item()),
        "accuracy": float(predictions.eq(test_labels).float().mean().item()),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "balanced_accuracy": float(recall),
        "worst_class_recall": float(min(per_class_recall)),
        "per_class_recall": [float(value) for value in per_class_recall],
        "per_class_f1": [float(value) for value in per_class_f1],
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(num_classes))
        ).tolist(),
        "num_samples": int(len(test_dataset)),
        "uncalibrated": probability_metrics(test_logits, test_labels),
    }
    if threshold is not None:
        result.update(
            {
                "calibration_num_samples": len(calibration_indices),
                "temperature": temperature,
                "calibrated": probability_metrics(
                    test_logits, test_labels, temperature=temperature
                ),
                "conformal_alpha": conformal_alpha,
                "conformal_threshold": threshold,
                "conformal": conformal_metrics(
                    test_logits, test_labels, threshold, temperature
                ),
            }
        )
    return result


def run_one(
    num_clients: int,
    alpha: float,
    strategy_name: str,
    device: torch.device,
    num_rounds: int = NUM_ROUNDS,
    local_epochs: int = LOCAL_EPOCHS,
    batch_size: int = 32,
    client_cpus: float = None,
    client_gpus: float = None,
    train_samples: int = 5000,
    calibration_fraction: float = 0.5,
    conformal_alpha: float = 0.1,
    logit_tau: float = 1.0,
    prior_smoothing: float = 1.0,
    head_mu: float = 0.01,
    coverage_kappa: float = 32.0,
    seed: int = DEFAULT_SEED,
    size: int = 28,
    model_name: str = "legacy",
    augment: bool = False,
    normalization_mode: str = "fixed",
    final_test: bool = False,
    locked_config: str = None,
    early_stop_patience: int = 0,
) -> dict:

    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    if final_test and not locked_config:
        raise ValueError("final test requires --locked_config from a development run")

    print(f"\n{'='*65}")
    print(f"  FL | clients={num_clients} | alpha={alpha}"
          f" | strategy={strategy_name.upper()}")
    print(f"{'='*65}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    resources = dict(CLIENT_RESOURCES)
    if client_cpus is not None:
        resources["num_cpus"] = client_cpus
    if client_gpus is not None:
        resources["num_gpus"] = client_gpus
    if not torch.cuda.is_available():
        resources["num_gpus"] = 0.0
    client_device = torch.device(
        "cuda" if resources.get("num_gpus", 0.0) > 0 else "cpu"
    )

    # 1. Dataset
    train_ds, num_classes = get_bloodmnist_dataset("train", download=True, size=size)

    # 2. Apply an explicit 5K budget before non-IID client partitioning.
    eligible_train_indices = stratified_subsample_indices(
        train_ds, num_samples=train_samples, seed=seed
    )
    normalization = None
    if normalization_mode == "train":
        normalization = fit_train_normalization(train_ds, eligible_train_indices)
    train_ds.transform = build_transform("train", augment=augment, normalization=normalization)
    val_ds, _ = get_bloodmnist_dataset(
        "val", download=True, size=size, normalization=normalization
    )
    partition = dirichlet_partition(
        train_ds,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
        eligible_indices=eligible_train_indices,
    )
    local_val_indices, calibration_indices = stratified_holdout_indices(
        val_ds, holdout_fraction=calibration_fraction, seed=seed
    )
    val_partition = stratified_validation_partition(
        val_ds,
        num_clients=num_clients,
        seed=seed,
        eligible_indices=local_val_indices,
    )
    if any(not indices for indices in partition):
        raise ValueError(
            "At least one client has no training samples; use fewer clients "
            "or a larger Dirichlet alpha."
        )

    # 3. Model goc
    torch.manual_seed(seed)
    init_model  = build_model(model_name, num_classes=num_classes)
    model_bytes = calc_model_size_bytes(init_model)
    model_kb    = model_bytes / 1024
    init_params = fl.common.ndarrays_to_parameters(get_parameters(init_model))
    print(f"  Model size: {model_kb:.1f} KB")

    # 4. Strategy
    strategy = get_strategy(strategy_name, num_clients, init_params)
    strategy.latest_parameters = init_params

    # 5. Client factory
    client_fn = make_client_fn(
        train_partition=partition,
        val_partition=val_partition,
        local_epochs=local_epochs,
        lr=LR,
        device=client_device,
        num_classes=num_classes,
        batch_size=batch_size,
        model_name=model_name,
        seed=seed,
        size=size,
        augment=augment,
        normalization=normalization,
    )

    # 6. Callbacks (dung CFG thay vi hardcode)
    early_stopper = EarlyStoppingCallback(
        patience=early_stop_patience, min_delta=ES_MIN_DELTA)
    lr_scheduler  = GlobalLRScheduler(
        initial_lr=LR, factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=LR_MIN)

    history_acc  = []
    round_times  = []
    round_start  = [time.time()]
    actual_rounds = [0]

    # 7. Monkey-patch callbacks
    orig_agg = strategy.aggregate_evaluate
    orig_agg_fit = strategy.aggregate_fit

    def agg_fit_with_tracking(server_round, results, failures):
        aggregated = orig_agg_fit(server_round, results, failures)
        if aggregated is not None and aggregated[0] is not None:
            strategy.latest_parameters = aggregated[0]
        return aggregated

    def agg_with_tracking(server_round, results, failures):
        t_round = time.time() - round_start[0]
        round_times.append(t_round)
        round_start[0] = time.time()
        actual_rounds[0] = server_round

        agg = orig_agg(server_round, results, failures)
        if agg is not None:
            _, metrics_agg = agg
            acc = metrics_agg.get("accuracy",  0.0)
            pre = metrics_agg.get("precision", 0.0)
            rec = metrics_agg.get("recall",    0.0)
            f1  = metrics_agg.get("f1_score",  0.0)
            history_acc.append(acc)
            new_lr = lr_scheduler.step(acc)
            print(f"  [Round {server_round:3d}] "
                  f"Acc={acc:.4f} Pre={pre:.4f} "
                  f"Rec={rec:.4f} F1={f1:.4f} "
                  f"LR={new_lr:.6f} t={t_round:.1f}s")
            if early_stop_patience > 0 and not early_stopper.update(server_round, acc):
                raise EarlyStopException()
        return agg

    strategy.aggregate_evaluate = agg_with_tracking
    strategy.aggregate_fit = agg_fit_with_tracking
    def fit_config(_round):
        config = lr_scheduler.get_config()
        config["server_round"] = _round
        config["balanced_sampling"] = strategy_name == "balanced"
        if strategy_name in {"coverage", "logit_only", "head_only"}:
            config.update(
                {
                    "logit_tau": logit_tau if strategy_name in {"coverage", "logit_only"} else 0.0,
                    "prior_smoothing": prior_smoothing,
                    "head_mu": head_mu if strategy_name in {"coverage", "head_only"} else 0.0,
                    "coverage_kappa": coverage_kappa,
                }
            )
        return config

    strategy.on_fit_config_fn = fit_config

    # 8. Chay
    t0 = time.time()
    try:
        print(f"  Virtual client resources: {resources}")

        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources=resources,
        )
    except Exception as e:
        # Flower sẽ bọc lỗi của chúng ta vào trong e.__cause__
        if isinstance(e.__cause__, EarlyStopException):
            print("\n  >> Da chu dong ngat Flower Simulation vi kich hoat Early Stopping!")
        else:
            raise e  # Nếu là lỗi thật sự (code sai, hết RAM...) thì văng ra để biết
    total_time     = time.time() - t0
    if strategy.latest_parameters is init_params:
        raise RuntimeError("No client model updates were aggregated; inspect Flower client failures")
    actual_n       = actual_rounds[0]
    from monitoring.research_validation import evaluate_validation
    final_validation_metrics = evaluate_validation(
        parameters_to_ndarrays(strategy.latest_parameters), model_name,
        val_ds, local_val_indices, num_classes=num_classes,
        batch_size=batch_size,
    )
    final_test_metrics = None
    if final_test:
        final_test_metrics = evaluate_final_global_model(
            strategy.latest_parameters,
            num_classes,
            device,
            batch_size,
            calibration_indices=calibration_indices,
            conformal_alpha=conformal_alpha,
            model_name=model_name,
            size=size,
            normalization=normalization,
        )

    # 9. Communication
    bytes_per_round   = 2 * model_bytes * num_clients
    total_comm_mb     = (bytes_per_round * actual_n) / (1024 ** 2)
    comm_per_round_kb = bytes_per_round / 1024
    convergence_round = (history_acc.index(max(history_acc)) + 1
                         if history_acc else 0)

    # 10. Luu
    result = {
        "strategy": strategy_name, "num_clients": num_clients, "alpha": alpha,
        "model": model_name, "size": size, "augment": augment,
        "normalization": normalization_mode, "normalization_stats": normalization,
        "seed": seed, "final_test_enabled": final_test,
        "client_resources": resources,
        "local_epochs": local_epochs, "fl_rounds_target": num_rounds,
        "early_stop_patience": early_stop_patience,
        "fl_rounds_actual": actual_n,
        "best_accuracy": float(max(history_acc)) if history_acc else 0.0,
        "final_accuracy": float(history_acc[-1]) if history_acc else 0.0,
        "convergence_round": convergence_round,
        "accuracy_history": history_acc,
        "total_time_s": round(total_time, 1),
        "avg_time_per_round_s": round(
            sum(round_times) / len(round_times) if round_times else 0, 2),
        "model_size_kb": round(model_kb, 2),
        "comm_per_round_kb": round(comm_per_round_kb, 2),
        "total_comm_mb": round(total_comm_mb, 3),
        "final_test_metrics": final_test_metrics,
        "final_validation_metrics": final_validation_metrics,
        "val_f1_macro": final_validation_metrics["f1_macro"],
        "val_worst_class_recall": final_validation_metrics["worst_class_recall"],
        "train_samples_requested": train_samples,
        "train_samples_used": len(eligible_train_indices),
        "local_validation_samples": len(local_val_indices),
        "calibration_samples": len(calibration_indices),
        "calibration_fraction": calibration_fraction,
        "conformal_alpha": conformal_alpha,
        "coverage_objective": {
            "enabled": strategy_name in {"coverage", "logit_only", "head_only"},
            "logit_tau": logit_tau if strategy_name in {"coverage", "logit_only"} else 0.0,
            "prior_smoothing": prior_smoothing,
            "head_mu": head_mu if strategy_name in {"coverage", "head_only"} else 0.0,
            "coverage_kappa": coverage_kappa,
        },
        "train_partition_sizes": [len(indices) for indices in partition],
        "train_partition_hashes": [
            _partition_hash(indices) for indices in partition
        ],
        "local_validation_hash": _partition_hash(local_val_indices),
        "calibration_hash": _partition_hash(calibration_indices),
        # --- GPU metrics (moi them) ---
        **(get_gpu_usage() if client_device.type == "cuda" else
           {"peak_gpu_memory_mb": 0.0, "current_gpu_memory_mb": 0.0}),
    }

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_dir = RESULTS_DIR / strategy_name / f"clients{num_clients}_alpha{alpha}_seed{seed}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    if final_test_metrics is not None:
        with open(save_dir / "final_test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(final_test_metrics, f, indent=4)
    else:
        run_spec = {
            "num_clients": num_clients, "alpha": alpha, "strategy": strategy_name,
            "rounds": num_rounds, "local_epochs": local_epochs, "batch_size": batch_size,
            "client_cpus": resources["num_cpus"], "client_gpus": resources["num_gpus"],
            "train_samples": train_samples, "calibration_fraction": calibration_fraction,
            "conformal_alpha": conformal_alpha, "logit_tau": logit_tau,
            "prior_smoothing": prior_smoothing, "head_mu": head_mu,
            "coverage_kappa": coverage_kappa, "seed": seed, "size": size,
            "model": model_name, "augment": augment, "normalization": normalization_mode,
            "early_stop_patience": early_stop_patience,
        }
        with open(save_dir / "run_spec.json", "w", encoding="utf-8") as f:
            json.dump(run_spec, f, indent=2)

    print(f"\n  >> BestAcc={result['best_accuracy']:.4f} | "
          f"Round={convergence_round} | "
          f"Time={total_time:.0f}s | Comm={total_comm_mb:.1f}MB | "
          f"Mode={'final' if final_test else 'development'}")
    return result


def generate_comparison_table(all_results: list):
    """In bang so sanh va luu CSV day du sau khi chay xong tat ca thi nghiem."""
    if not all_results:
        return

    # Kiem tra co GPU khong de quyet dinh hien thi cot PeakGPU
    has_gpu = any(r.get("peak_gpu_memory_mb", 0) > 0 for r in all_results)

    # --- In terminal ---
    if has_gpu:
        header = (f"{'Strategy':<20} {'Clients':>7} {'Alpha':>6} "
                  f"{'BestAcc':>8} {'FinalAcc':>9} {'ConvRnd':>8} "
                  f"{'Rounds':>7} {'Time(s)':>8} "
                  f"{'ModelKB':>8} {'CommMB':>7} {'ValF1':>7} {'WorstRec':>8} {'PeakGPU(MB)':>12}")
    else:
        header = (f"{'Strategy':<20} {'Clients':>7} {'Alpha':>6} "
                  f"{'BestAcc':>8} {'FinalAcc':>9} {'ConvRnd':>8} "
                  f"{'Rounds':>7} {'Time(s)':>8} "
                  f"{'ModelKB':>8} {'CommMB':>7} {'ValF1':>7} {'WorstRec':>8} {'PeakGPU(MB)':>12}")

    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print("  VALIDATION COMPARISON (not a centralized or final-test score)")
    print(f"{'='*len(header)}\n{header}\n{sep}")

    for r in sorted(all_results, key=lambda x: x["val_f1_macro"], reverse=True):
        peak_gpu = r.get("peak_gpu_memory_mb", 0.0)
        print(f"{r['strategy']:<20} "
              f"{r['num_clients']:>7} "
              f"{r['alpha']:>6.1f} "
              f"{r['best_accuracy']:>8.4f} "
              f"{r['final_accuracy']:>9.4f} "
              f"{r['convergence_round']:>8} "
              f"{r['fl_rounds_actual']:>7} "
              f"{r['total_time_s']:>8.1f} "
              f"{r['model_size_kb']:>8.1f} "
              f"{r['total_comm_mb']:>7.1f} "
              f"{r['val_f1_macro']:>7.4f} "
              f"{r['val_worst_class_recall']:>8.4f} "
              f"{peak_gpu:>12.1f}")
    print(sep)

    # --- Luu CSV ---
    # Tat ca cac truong scalar tu result dict, bo qua accuracy_history (la list)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    keys = [
        # Tham so thi nghiem
        "strategy",
        "num_clients",
        "alpha",
        "local_epochs",
        "fl_rounds_target",
        "fl_rounds_actual",
        # Hieu nang
        "best_accuracy",
        "final_accuracy",
        "val_f1_macro",
        "val_worst_class_recall",
        "convergence_round",
        # Thoi gian
        "total_time_s",
        "avg_time_per_round_s",
        # Giao tiep
        "model_size_kb",
        "comm_per_round_kb",
        "total_comm_mb",
        # GPU
        "peak_gpu_memory_mb",
        "current_gpu_memory_mb",
    ]

    csv_path = RESULTS_DIR / "comparison_table.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in all_results:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    print(f"\n  CSV da luu: {csv_path}")
    print(f"  Tong so thi nghiem: {len(all_results)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int,   default=None)
    parser.add_argument("--alpha",       type=float, default=None)
    parser.add_argument("--strategy",    type=str,   default=None)
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--local_epochs", type=int, default=LOCAL_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--client_cpus", type=float, default=None)
    parser.add_argument("--client_gpus", type=float, default=None)
    parser.add_argument("--train_samples", type=int, default=5000)
    parser.add_argument("--calibration_fraction", type=float, default=0.5)
    parser.add_argument("--conformal_alpha", type=float, default=0.1)
    parser.add_argument("--logit_tau", type=float, default=1.0)
    parser.add_argument("--prior_smoothing", type=float, default=1.0)
    parser.add_argument("--head_mu", type=float, default=0.01)
    parser.add_argument("--coverage_kappa", type=float, default=32.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--size", type=int, choices=[28, 64], default=28)
    parser.add_argument("--model", choices=["legacy", "tiny_cnn", "mobilenet_v3_small"], default="legacy")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--normalization", choices=["fixed", "train"], default="fixed")
    parser.add_argument("--final_test", action="store_true")
    parser.add_argument("--locked_config", default=None)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--all",  action="store_true")
    args = parser.parse_args()
    if args.locked_config:
        if not args.final_test:
            parser.error("--locked_config requires --final_test")
        with open(args.locked_config, encoding="utf-8") as f:
            locked_spec = json.load(f)
        for key, value in locked_spec.items():
            if not hasattr(args, key) or key in {"all", "final_test", "locked_config"}:
                parser.error(f"invalid locked setting: {key}")
            setattr(args, key, value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_info = get_gpu_info()
    print(f"Device  : {device}")
    print(f"GPU     : {gpu_info['gpu_name']}")
    if gpu_info["gpu_available"]:
        print(f"VRAM    : {gpu_info['gpu_total_memory_gb']} GB")
        print(f"CUDA    : {gpu_info['cuda_version']}")
    print(f"Config  : {_CFG_PATH}")
    # Luu gpu_info vao 1 file chung
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "gpu_info.json", "w") as f:
        json.dump(gpu_info, f, indent=4)


    all_results = []
    if args.all:
        total = len(DEFAULT_NUM_CLIENTS)*len(DEFAULT_ALPHAS)*len(DEFAULT_STRATEGIES)
        print(f"Chay {total} thi nghiem...")
        for nc in DEFAULT_NUM_CLIENTS:
            for al in DEFAULT_ALPHAS:
                for st in DEFAULT_STRATEGIES:
                    r = run_one(
                        nc, al, st, device,
                        num_rounds=args.rounds,
                        local_epochs=args.local_epochs,
                        batch_size=args.batch_size,
                        client_cpus=args.client_cpus,
                        client_gpus=args.client_gpus,
                        train_samples=args.train_samples,
                        calibration_fraction=args.calibration_fraction,
                        conformal_alpha=args.conformal_alpha,
                        logit_tau=args.logit_tau,
                        prior_smoothing=args.prior_smoothing,
                        head_mu=args.head_mu,
                        coverage_kappa=args.coverage_kappa,
                        seed=args.seed, size=args.size, model_name=args.model,
                        augment=args.augment, normalization_mode=args.normalization,
                        final_test=args.final_test, locked_config=args.locked_config,
                        early_stop_patience=args.early_stop_patience,
                    )
                    if r: all_results.append(r)
    else:
        r = run_one(
            args.num_clients or 5,
            args.alpha or 0.3,
            args.strategy or "fedavg",
            device,
            num_rounds=args.rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            client_cpus=args.client_cpus,
            client_gpus=args.client_gpus,
            train_samples=args.train_samples,
            calibration_fraction=args.calibration_fraction,
            conformal_alpha=args.conformal_alpha,
            logit_tau=args.logit_tau,
            prior_smoothing=args.prior_smoothing,
            head_mu=args.head_mu,
            coverage_kappa=args.coverage_kappa,
            seed=args.seed, size=args.size, model_name=args.model,
            augment=args.augment, normalization_mode=args.normalization,
            final_test=args.final_test, locked_config=args.locked_config,
            early_stop_patience=args.early_stop_patience,
        )
        if r: all_results.append(r)

    generate_comparison_table(all_results)
    print("\nHoan thanh!")

if __name__ == "__main__":
    main()
