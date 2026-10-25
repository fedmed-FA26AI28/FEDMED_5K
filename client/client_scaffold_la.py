"""
Flower Client cho SCAFFOLD + Logit Adjustment (SCAFFOLD-LA).
Kết hợp Control Variates của SCAFFOLD với Logit Adjustment margin.
"""

import pickle
from pathlib import Path
import numpy as np
import torch
import flwr as fl
from torch.utils.data import DataLoader, Subset
from flwr.common import Context

from models.cnn import SimpleCNN, get_parameters, set_parameters
from models.cnn_570k import CNN570K
from client.train_scaffold_la import train_scaffold_la
from client.evaluate import evaluate

SCAFFOLD_STATE_DIR = Path("data/scaffold_states")
SCAFFOLD_STATE_DIR.mkdir(parents=True, exist_ok=True)


def _make_optimizer(net, optimizer_type, lr, momentum=0.0, weight_decay=0.0):
    if optimizer_type == "sgd":
        return torch.optim.SGD(
            net.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    return torch.optim.Adam(net.parameters(), lr=lr)


class ScaffoldLAFlowerClient(fl.client.NumPyClient):
    """
    Flower Client triển khai thuật toán SCAFFOLD-LA.
    """
    def __init__(self, client_id: int, net, train_loader, val_loader,
                 test_loader_local, test_loader_global,
                 local_epochs, lr, device, log_priors=None, tau_la=1.0,
                 optimizer_type="sgd", momentum=0.9, weight_decay=1e-4):
        self.client_id          = client_id
        self.net                = net
        self.train_loader       = train_loader
        self.val_loader         = val_loader
        self.test_loader_local  = test_loader_local
        self.test_loader_global = test_loader_global
        self.local_epochs       = local_epochs
        self.lr                 = lr
        self.device             = device
        self.log_priors         = log_priors
        self.tau_la             = tau_la
        self.optimizer_type     = optimizer_type
        self.momentum           = momentum
        self.weight_decay       = weight_decay

    def _load_c_local(self):
        state_file = SCAFFOLD_STATE_DIR / f"c_local_{self.client_id}.pkl"
        if state_file.exists():
            try:
                with open(state_file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return [torch.zeros_like(p.data.cpu()) for p in self.net.parameters()]

    def _load_c_global(self):
        state_file = SCAFFOLD_STATE_DIR / "c_global.pkl"
        if state_file.exists():
            try:
                with open(state_file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return [torch.zeros_like(p.data.cpu()) for p in self.net.parameters()]

    def _save_c_local(self, c_local_new):
        state_file = SCAFFOLD_STATE_DIR / f"c_local_{self.client_id}.pkl"
        with open(state_file, "wb") as f:
            pickle.dump(c_local_new, f)

    def _save_delta_c(self, delta_c):
        state_file = SCAFFOLD_STATE_DIR / f"delta_c_{self.client_id}.pkl"
        with open(state_file, "wb") as f:
            pickle.dump(delta_c, f)

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)

        current_lr = config.get("lr", self.lr)
        tau_la = float(config.get("tau_la", self.tau_la))

        optimizer = _make_optimizer(
            self.net, self.optimizer_type, current_lr, self.momentum, self.weight_decay
        )

        c_global = self._load_c_global()
        c_local  = self._load_c_local()

        result = train_scaffold_la(
            model=self.net,
            train_loader=self.train_loader,
            optimizer=optimizer,
            epochs=self.local_epochs,
            device=self.device,
            c_global=c_global,
            c_local=c_local,
            lr=current_lr,
            log_priors=self.log_priors,
            tau_la=tau_la,
        )

        # Lưu trạng thái mới
        self._save_c_local(result["c_local_new"])
        self._save_delta_c(result["delta_c"])

        fit_metrics = {
            "client_id": self.client_id,
            "train_loss": float(result.get("final_loss", 0.0)),
            "train_accuracy": float(result.get("final_accuracy", 0.0)),
            "tau": float(result["tau"]),
            "num_samples": float(result["num_samples"]),
        }
        return get_parameters(self.net), len(self.train_loader.dataset), fit_metrics

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        # 1. Đánh giá cá nhân hóa trên test set cục bộ của Client (BMCTA)
        loss_local, eval_metrics_local = evaluate(self.net, self.test_loader_local, self.device)

        # 2. Đánh giá toàn cục trên full test set (BTA)
        loss_global, eval_metrics_global = evaluate(self.net, self.test_loader_global, self.device)

        return loss_local, len(self.test_loader_local.dataset), {
            "accuracy":        float(eval_metrics_local["accuracy"]),   # Local Client Acc (BMCTA)
            "global_accuracy": float(eval_metrics_global["accuracy"]),  # Global Test Acc (BTA)
            "precision":       float(eval_metrics_local["precision"]),
            "recall":          float(eval_metrics_local["recall"]),
            "f1_score":        float(eval_metrics_local["f1_score"]),
        }


def make_scaffold_la_client_fn(train_dataset, val_dataset, test_dataset,
                               partition, local_epochs, lr, device,
                               num_classes=11, batch_size=256,
                               optimizer_type="sgd", momentum=0.9,
                               weight_decay=1e-4, model_type="570k",
                               tau_la=1.0, test_partition=None):
    """
    Factory function trả về hàm client_fn cho SCAFFOLD-LA trong Flower Simulation.
    Tính toán phân phối xác suất tiên nghiệm (class priors) cho từng client từ partition.
    """
    def client_fn(context: Context) -> fl.client.Client:
        client_id = int(context.node_config["partition-id"])
        client_indices = partition[client_id]

        train_subset = Subset(train_dataset, client_indices)
        train_loader = DataLoader(train_subset, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=0)

        # Phân chia test loader: local subset vs full global
        if test_partition is not None and client_id < len(test_partition) and len(test_partition[client_id]) > 0:
            test_subset_local = Subset(test_dataset, test_partition[client_id])
            test_loader_local = DataLoader(test_subset_local, batch_size=batch_size,
                                           shuffle=False, num_workers=0)
        else:
            test_loader_local = DataLoader(test_dataset, batch_size=batch_size,
                                           shuffle=False, num_workers=0)

        test_loader_global = DataLoader(test_dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=0)

        # Tính class priors cho client i
        if hasattr(train_dataset, 'labels'):
            client_labels = train_dataset.labels[client_indices].reshape(-1)
            counts = np.bincount(client_labels, minlength=num_classes).astype(np.float32)
            counts = counts + 1e-4  # Laplace smoothing tránh log(0)
            priors = counts / counts.sum()
            log_priors = torch.tensor(np.log(priors), dtype=torch.float32)
        else:
            log_priors = None

        in_channels = train_dataset[0][0].shape[0]
        if model_type == "570k":
            net = CNN570K(in_channels=in_channels, num_classes=num_classes).to(device)
        else:
            net = SimpleCNN(in_channels=in_channels, num_classes=num_classes).to(device)

        return ScaffoldLAFlowerClient(
            client_id=client_id,
            net=net,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader_local=test_loader_local,
            test_loader_global=test_loader_global,
            local_epochs=local_epochs,
            lr=lr,
            device=device,
            log_priors=log_priors,
            tau_la=tau_la,
            optimizer_type=optimizer_type,
            momentum=momentum,
            weight_decay=weight_decay,
        ).to_client()

    return client_fn
