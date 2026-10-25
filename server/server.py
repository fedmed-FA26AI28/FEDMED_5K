"""Điểm vào (entry point) của FL Server. Khởi chạy server, chọn chiến lược (strategy) và điều phối các round FL."""
"""FL Server - Hỗ trợ các chiến lược chuẩn của Flower và các thuật toán nâng cao (FedNova, SCAFFOLD)."""

import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics
from flwr.server.strategy import (
    FedAvg, FedAvgM, FedProx,
    FedMedian, FedTrimmedAvg,
)
from algorithms.fednova_strategy import FedNova
from algorithms.scaffold_strategy import SCAFFOLD

# Danh sách đầy đủ chiến lược hỗ trợ
ALL_STRATEGIES = [
    "fedavg",
    "fedavg_la",
    "fedprox",
    "fedprox_la",
    "fedbn",
    "fednova",
    "scaffold",
    "scaffold_la",
    "fednova_scaffold",
    "fedbn_scaffold_la",
]

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Tự động tính trung bình có trọng số cho các metrics (accuracy, precision...) từ client."""
    total_examples = sum([num_examples for num_examples, _ in metrics])
    if total_examples == 0:
        return {}

    weighted_metrics = {}
    for num_examples, m in metrics:
        for key, value in m.items():
            if key not in weighted_metrics:
                weighted_metrics[key] = 0.0
            weighted_metrics[key] += num_examples * float(value)
    # Chia cho tổng số data để ra trung bình
    return {k: v / total_examples for k, v in weighted_metrics.items()}

def get_strategy(strategy_name: str, num_clients: int,
                 initial_parameters,
                 proximal_mu: float = 0.01):
    """Trả về Flower Strategy theo tên."""

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

    if name in ["fedavg", "fedavg_la"]:
        return FedAvg(**base)

    elif name in ["fedprox", "fedprox_la"]:
        return FedProx(**base, proximal_mu=proximal_mu)

    elif name == "fedbn":
        return FedAvg(**base)

    elif name == "fednova":
        return FedNova(**base)

    elif name in ["scaffold", "scaffold_la", "fedbn_scaffold_la"]:
        return SCAFFOLD(num_clients=num_clients, **base)

    elif name == "fednova_scaffold":
        from algorithms.fednova_scaffold_strategy import FedNovaScaffold
        return FedNovaScaffold(num_clients=num_clients, **base)

    else:
        raise ValueError(
            f"Strategy '{strategy_name}' không hợp lệ.\n"
            f"Chọn trong: {ALL_STRATEGIES}"
        )


class EarlyStoppingCallback:
    """Global Early Stopping dựa trên Global Accuracy."""
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
                print(f"\n[Global Early Stop] Dừng tại Round {current_round}.")
                print(f"  Best Acc: {self.best_acc:.4f} tại Round {self.best_round}")
                self.stop = True
        return not self.stop


class GlobalLRScheduler:
    """
    Giảm Learning Rate toàn cục khi Global Accuracy không tăng.
    Server gửi LR mới xuống Client thông qua config dict trong mỗi Round.
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
        if accuracy > self.best_acc + 1e-4:
            self.best_acc = accuracy
            self.wait     = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                new_lr = max(self.lr * self.factor, self.min_lr)
                if new_lr < self.lr:
                    print(f"  [Global LR] Giảm LR: {self.lr:.6f} -> {new_lr:.6f}")
                    self.lr = new_lr
                self.wait = 0

        return self.lr

    def get_config(self) -> dict:
        return {"lr": self.lr}
