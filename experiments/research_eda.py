"""Train-only BloodMNIST EDA; deliberately never instantiate val or test."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from medmnist import INFO

from datasets.medmnist_code import get_bloodmnist_dataset
from datasets.partition import dirichlet_partition, stratified_subsample_indices


def run_eda(size=64, train_samples=5000, num_clients=10, alpha=0.1,
            seed=42, output_dir="results/research_eda"):
    dataset, num_classes = get_bloodmnist_dataset("train", download=True, size=size)
    selected = stratified_subsample_indices(dataset, train_samples, seed=seed)
    parts = dirichlet_partition(dataset, num_clients, alpha, seed=seed,
                                eligible_indices=selected)
    labels = np.asarray(dataset.labels).reshape(-1).astype(int)
    matrix = np.zeros((num_clients, num_classes), dtype=int)
    for client_id, part in enumerate(parts):
        matrix[client_id] = np.bincount(labels[part], minlength=num_classes)

    rgb_means = np.empty((len(selected), 3), dtype=np.float64)
    for start in range(0, len(selected), 256):
        ids = selected[start:start + 256]
        batch = np.asarray(dataset.imgs[ids], dtype=np.float32) / 255.0
        rgb_means[start:start + len(ids)] = batch.mean(axis=(1, 2))
    brightness = rgb_means.mean(axis=1)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for class_id in range(num_classes):
        mask = labels[selected] == class_id
        class_rgb = rgb_means[mask]
        rows.append({
            "class_id": class_id,
            "name": INFO["bloodmnist"]["label"][str(class_id)],
            "count": int(mask.sum()),
            "brightness_mean": float(brightness[mask].mean()),
            "brightness_std": float(brightness[mask].std()),
            "red_mean": float(class_rgb[:, 0].mean()),
            "green_mean": float(class_rgb[:, 1].mean()),
            "blue_mean": float(class_rgb[:, 2].mean()),
        })
    with (output / "class_statistics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "split": "train", "size": size, "train_samples": len(selected),
        "num_clients": num_clients, "alpha": alpha, "seed": seed,
        "class_counts": [row["count"] for row in rows],
        "client_class_counts": matrix.tolist(),
        "client_sizes": matrix.sum(axis=1).tolist(),
        "classes_missing_per_client": (matrix == 0).sum(axis=1).tolist(),
        "brightness_mean": float(brightness.mean()),
        "brightness_std": float(brightness.std()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(range(num_classes), summary["class_counts"])
    axes[0].set(xlabel="Class", ylabel="Training samples", title="Train-only class counts")
    axes[1].imshow(matrix, aspect="auto", cmap="Blues")
    axes[1].set(xlabel="Class", ylabel="Virtual client", title="Client training coverage")
    fig.tight_layout()
    fig.savefig(output / "distribution.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(11, 6))
    for class_id, ax in enumerate(axes.flat):
        first = next(index for index in selected if labels[index] == class_id)
        ax.imshow(dataset.imgs[first])
        ax.set_title(rows[class_id]["name"])
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "train_gallery.png", dpi=150)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, choices=[28, 64], default=64)
    parser.add_argument("--train_samples", type=int, default=5000)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/research_eda")
    print(json.dumps(run_eda(**vars(parser.parse_args())), indent=2))
