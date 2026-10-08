"""Điểm vào (entry point) của FL Server. Khởi chạy server, chọn chiến lược (strategy) và điều phối các round FL."""
"""FL Server - Ho tro 11 strategies co san trong Flower."""

import flwr as fl
from typing import List, Tuple, Dict
from flwr.common import Metrics
from flwr.server.strategy import (
    FedAvg, FedAvgM, FedProx,
    FedMedian, FedTrimmedAvg,
    
)

# Danh sach day du de dung trong run_simulation.py
ALL_STRATEGIES = [
    "coverage",
    "balanced", "logit_only", "head_only",
    "fedavg",       # Baseline chuẩn
    "fedavgm",      # FedAvg + Momentum
    "fedprox",      # Tốt nhất cho Non-IID
    "fedmedian",    # Robust aggregation
    "fedtrimmedavg" # Robust aggregation
]

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Tự động tính trung bình có trọng số cho các metrics (accuracy, precision...) từ client."""
    total_examples = sum([num_examples for num_examples, _ in metrics])
    
    weighted_metrics = {}
    for num_examples, m in metrics:
        for key, value in m.items():
            if key not in weighted_metrics:
                weighted_metrics[key] = 0.0
            weighted_metrics[key] += num_examples * float(value)
    # Chia cho tổng số data để ra trung bình
    aggregated = {k: v / total_examples for k, v in weighted_metrics.items()}
    if metrics and all("accuracy" in values for _, values in metrics):
        aggregated["worst_client_accuracy"] = min(
            float(values["accuracy"]) for _, values in metrics
        )
    if metrics and all("f1_score" in values for _, values in metrics):
        aggregated["worst_client_f1_macro"] = min(
            float(values["f1_score"]) for _, values in metrics
        )
    return aggregated

def get_strategy(strategy_name: str, num_clients: int,
                 initial_parameters,
                 proximal_mu: float = 0.01):
    """Tra ve Flower Strategy theo ten."""

    fraction_fit     = 1.0
    fraction_eval    = 1.0
    min_fit_clients  = num_clients
    min_eval_clients = num_clients
    min_available    = num_clients

    base = dict(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_eval,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_eval_clients,
        min_available_clients=min_available,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average, 
        fit_metrics_aggregation_fn=weighted_average,
    )

    name = strategy_name.lower()

    if name in {"fedavg", "coverage", "balanced", "logit_only", "head_only"}:
        return FedAvg(**base)

    elif name == "fedprox":
        return FedProx(**base, proximal_mu=proximal_mu)
    
    else:
        raise ValueError(
            f"Strategy '{strategy_name}' khong hop le.\n"
            f"Chon trong: {ALL_STRATEGIES}"
        )


class EarlyStoppingCallback:
    """Global Early Stopping dua tren Global Accuracy."""
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_acc   = 0.0
        self.wait       = 0
        self.stop       = False
        self.best_round = 0

    def update(self, current_round: int, accuracy: float) -> bool:
        if accuracy > self.best_acc + self.min_delta:
            self.best_acc   = accuracy
            self.wait       = 0
            self.best_round = current_round
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"\n[Global Early Stop] Dung tai Round {current_round}.")
                print(f"  Best Acc: {self.best_acc:.4f} tai Round {self.best_round}")
                self.stop = True
        return not self.stop


class GlobalLRScheduler:
    """
    Giam Learning Rate toan cuc khi Global Accuracy khong tang.
    Server gui LR moi xuong Client thong qua config dict trong moi Round.
    
    Su dung:
        - Server truyen LR vao fit() config: {"lr": current_lr}
        - Client doc config["lr"] de override LR cuc bo
    """
    def __init__(self, initial_lr: float = 0.001,
                 factor: float = 0.5,
                 patience: int = 5,
                 min_lr: float = 1e-6):
        self.lr       = initial_lr
        self.factor   = factor
        self.patience = patience
        self.min_lr   = min_lr

        self.best_acc = 0.0
        self.wait     = 0

    def step(self, accuracy: float) -> float:
        """
        Kiem tra va giam LR neu can.
        Tra ve LR hien tai (co the da giam).
        """
        if accuracy > self.best_acc + 1e-4:
            self.best_acc = accuracy
            self.wait     = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                new_lr = max(self.lr * self.factor, self.min_lr)
                if new_lr < self.lr:
                    print(f"  [Global LR] Giam LR: {self.lr:.6f} -> {new_lr:.6f}")
                    self.lr = new_lr
                self.wait = 0

        return self.lr

    def get_config(self) -> dict:
        """Tra ve config dict de Server gui xuong Client."""
        return {"lr": self.lr}
