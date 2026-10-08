"""Benchmark one physical client's train shard and inference; never load test."""

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, Subset

from algorithms.coverage import (
    count_client_classes, coverage_head_penalty, logit_adjustment,
    snapshot_classifier_head,
)
from algorithms.vacant_distillation import (
    frozen_global_teacher, vacant_class_distillation_loss,
)
from datasets.medmnist_code import get_bloodmnist_dataset
from datasets.partition import dirichlet_partition, stratified_subsample_indices
from models.cnn import build_model, get_parameters


def benchmark(args):
    dataset, num_classes = get_bloodmnist_dataset("train", size=args.size, download=True)
    selected = stratified_subsample_indices(dataset, args.train_samples, args.seed)
    partition = dirichlet_partition(dataset, args.num_clients, args.alpha,
                                    seed=args.seed, eligible_indices=selected)
    indices = partition[args.client_id]
    if not indices:
        raise ValueError("selected client has no training samples")
    sampled = indices[:args.max_train_samples]
    loader = DataLoader(Subset(dataset, sampled), batch_size=args.batch_size,
                        shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    class_counts = count_client_classes(Subset(dataset, indices), num_classes)
    process = psutil.Process()
    peak_rss = [process.memory_info().rss]
    stop = threading.Event()

    def sample_memory():
        while not stop.wait(0.05):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    uses_coverage = args.strategy in {"coverage", "coverage_distill"}
    uses_distillation = args.strategy in {"vacant_distill", "coverage_distill"}
    global_head = snapshot_classifier_head(model) if uses_coverage else None
    adjustment = (logit_adjustment(class_counts, args.logit_tau).to(device)
                  if uses_coverage else None)
    teacher = frozen_global_teacher(
        model, class_counts, args.distill_mu if uses_distillation else 0.0,
        server_round=2, max_count=args.distill_max_count,
    )
    model.train()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.reshape(-1).long().to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(
            logits + adjustment if adjustment is not None else logits, labels
        )
        if global_head is not None:
            loss = loss + coverage_head_penalty(
                model, global_head, class_counts, args.head_mu, args.coverage_kappa,
            )
        if teacher is not None:
            with torch.no_grad():
                teacher_logits = teacher(images)
            loss = loss + args.distill_mu * vacant_class_distillation_loss(
                logits, teacher_logits, class_counts,
                temperature=args.distill_temperature,
                max_count=args.distill_max_count,
            )
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    model.eval()
    single_image = dataset[indices[0]][0].unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(args.warmup):
            model(single_image)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(args.repetitions):
            tick = time.perf_counter()
            model(single_image)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - tick) * 1000)
    stop.set()
    monitor.join(timeout=1)
    try:
        hardware = Path("/proc/device-tree/model").read_text().strip("\x00\n")
    except OSError:
        hardware = "non-Jetson or model unavailable"
    model_bytes = sum(array.nbytes for array in get_parameters(model))
    result = {
        "split": "train", "hardware": hardware, "device": str(device),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "model": args.model, "size": args.size, "strategy": args.strategy,
        "distill_active_classes": int((class_counts <= args.distill_max_count).sum().item()) if teacher is not None else 0,
        "teacher_model_copy": teacher is not None,
        "distill_mu": args.distill_mu if uses_distillation else 0.0,
        "distill_temperature": args.distill_temperature if uses_distillation else None,
        "train_samples_budget": len(selected),
        "client_samples": len(indices), "benchmarked_train_samples": len(sampled),
        "local_epoch_seconds": train_seconds,
        "train_samples_per_second": len(sampled) / train_seconds,
        "inference_batch_size": 1, "inference_warmup": args.warmup,
        "inference_repetitions": args.repetitions,
        "inference_p50_ms": float(np.percentile(times, 50)),
        "inference_p95_ms": float(np.percentile(times, 95)),
        "peak_process_rss_mb": peak_rss[0] / (1024 ** 2),
        "peak_torch_cuda_mb": (torch.cuda.max_memory_allocated() / (1024 ** 2)
                                if device.type == "cuda" else None),
        "model_bytes": model_bytes, "upload_bytes_per_round": model_bytes,
        "download_bytes_per_round": model_bytes,
        "communication_estimate_excludes_transport_overhead": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["legacy", "tiny_cnn", "mobilenet_v3_small"], default="tiny_cnn")
    parser.add_argument("--strategy", choices=["fedavg", "coverage", "vacant_distill", "coverage_distill"], default="fedavg")
    parser.add_argument("--distill_mu", type=float, default=0.1)
    parser.add_argument("--distill_temperature", type=float, default=2.0)
    parser.add_argument("--distill_max_count", type=int, default=0)
    parser.add_argument("--logit_tau", type=float, default=1.0)
    parser.add_argument("--head_mu", type=float, default=0.01)
    parser.add_argument("--coverage_kappa", type=float, default=32.0)
    parser.add_argument("--size", type=int, choices=[28, 64], default=64)
    parser.add_argument("--train_samples", type=int, default=5000)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--client_id", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_train_samples", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--output", default="results/jetson_benchmark/metrics.json")
    print(json.dumps(benchmark(parser.parse_args()), indent=2))
