"""Validation-only centralized comparator for the federated research runs.

This intentionally does not import or load the official test split. The old
``train_centralized`` script uses a different protocol and evaluates test.
"""

import argparse
import datetime
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.medmnist_code import (
    build_transform, fit_train_normalization, get_bloodmnist_dataset,
)
from datasets.partition import stratified_holdout_indices, stratified_subsample_indices
from models.cnn import build_model, get_parameters
from monitoring.research_validation import evaluate_validation


def _hash_indices(indices):
    ordered = np.asarray(sorted(indices), dtype=np.int64)
    return hashlib.sha256(ordered.tobytes()).hexdigest()


def run(args):
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    if args.final_test and not args.locked_config:
        raise ValueError("final test requires --locked_config from a development run")
    if not 0 < args.calibration_fraction < 1:
        raise ValueError("calibration_fraction must be in (0, 1)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(max(1, args.threads))

    train_dataset, num_classes = get_bloodmnist_dataset(
        "train", download=True, size=args.size,
    )
    train_indices = stratified_subsample_indices(
        train_dataset, num_samples=args.train_samples, seed=args.seed,
    )
    normalization = (
        fit_train_normalization(train_dataset, train_indices)
        if args.normalization == "train" else None
    )
    train_dataset.transform = build_transform(
        "train", augment=args.augment, normalization=normalization,
    )
    val_dataset, _ = get_bloodmnist_dataset(
        "val", download=True, size=args.size, normalization=normalization,
    )
    monitor_indices, calibration_indices = stratified_holdout_indices(
        val_dataset, holdout_fraction=args.calibration_fraction, seed=args.seed,
    )
    train_loader = DataLoader(
        Subset(train_dataset, train_indices), batch_size=args.batch_size,
        shuffle=True, num_workers=0,
    )
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    if args.use_gpu and device.type != "cuda":
        raise RuntimeError("--use_gpu requested, but CUDA is unavailable")
    # Match the FL initial model even if data preparation consumed RNG state.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = build_model(args.model, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_correct = total_seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.reshape(-1).long().to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.numel()
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_seen += labels.numel()
        validation = evaluate_validation(
            get_parameters(model), args.model, val_dataset, monitor_indices,
            num_classes=num_classes, batch_size=args.batch_size,
        )
        row = {
            "epoch": epoch, "train_loss": total_loss / total_seen,
            "train_accuracy": total_correct / total_seen,
            "val_f1_macro": validation["f1_macro"],
            "val_worst_class_recall": validation["worst_class_recall"],
            "val_accuracy": validation["accuracy"],
        }
        history.append(row)
        print(f"Epoch {epoch}/{args.epochs}: val_macro_f1={row['val_f1_macro']:.4f} "
              f"val_worst_recall={row['val_worst_class_recall']:.4f}", flush=True)

    final_test_metrics = None
    if args.final_test:
        # Import only in the locked final path. Development never loads test.
        from flwr.common import ndarrays_to_parameters
        from experiments.run_simulation import evaluate_final_global_model
        final_test_metrics = evaluate_final_global_model(
            ndarrays_to_parameters(get_parameters(model)), num_classes,
            device, args.batch_size, calibration_indices=calibration_indices,
            conformal_alpha=args.conformal_alpha, model_name=args.model,
            size=args.size, normalization=normalization,
        )
    result = {
        "mode": "centralized_final" if args.final_test else "centralized_development",
        "final_test_enabled": args.final_test,
        "model": args.model, "size": args.size, "augment": args.augment,
        "normalization": args.normalization, "normalization_stats": normalization,
        "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "train_samples_used": len(train_indices),
        "train_pool_hash": _hash_indices(train_indices),
        "local_validation_hash": _hash_indices(monitor_indices),
        "local_validation_samples": len(monitor_indices),
        "calibration_samples": len(calibration_indices),
        "calibration_fraction": args.calibration_fraction,
        "conformal_alpha": args.conformal_alpha,
        "device": str(device), "elapsed_seconds": round(time.time() - started, 3),
        "history": history, "final_validation_metrics": validation,
        "final_test_metrics": final_test_metrics,
    }
    output = Path(args.output_dir) / (
        f"seed{args.seed}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    output.mkdir(parents=True, exist_ok=False)
    with open(output / "results.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    if args.final_test:
        with open(output / "final_test_metrics.json", "w", encoding="utf-8") as file:
            json.dump(final_test_metrics, file, indent=2)
    else:
        spec = {key: value for key, value in vars(args).items()
                if key not in {"final_test", "locked_config", "output_dir"}}
        with open(output / "run_spec.json", "w", encoding="utf-8") as file:
            json.dump(spec, file, indent=2)
    print(f"Results: {output}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--train_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, choices=[28, 64], default=64)
    parser.add_argument("--model", choices=["tiny_cnn", "mobilenet_v3_small"], default="tiny_cnn")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--normalization", choices=["fixed", "train"], default="fixed")
    parser.add_argument("--calibration_fraction", type=float, default=0.5)
    parser.add_argument("--conformal_alpha", type=float, default=0.1)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output_dir", default="results/centralized_research")
    parser.add_argument("--final_test", action="store_true")
    parser.add_argument("--locked_config", default=None)
    args = parser.parse_args()
    if args.locked_config:
        if not args.final_test:
            parser.error("--locked_config requires --final_test")
        with open(args.locked_config, encoding="utf-8") as file:
            spec = json.load(file)
        for key, value in spec.items():
            if not hasattr(args, key) or key in {"final_test", "locked_config", "output_dir"}:
                parser.error(f"invalid locked setting: {key}")
            setattr(args, key, value)
    run(args)


if __name__ == "__main__":
    main()
