"""
Flower Client cho SCAFFOLD (Stochastic Controlled Averaging).
Duy trì trạng thái Control Variates cục bộ trên disk (data/scaffold_states/)
để đảm bảo tính liên tục giữa các rounds trong Flower Virtual Client Engine (Ray).
"""

import pickle
from pathlib import Path
import torch
import flwr as fl
from torch.utils.data import DataLoader, Subset
from flwr.common import Context

from models.cnn import SimpleCNN, get_parameters, set_parameters
from models.cnn_570k import CNN570K
from client.train_scaffold import train_scaffold
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


class ScaffoldFlowerClient(fl.client.NumPyClient):
    """
    Flower Client triển khai thuật toán SCAFFOLD.
    """
    def __init__(self, client_id: int, net, train_loader, val_loader, test_loader,
                 local_epochs, lr, device, optimizer_type="sgd",
                 momentum=0.9, weight_decay=1e-4):
        self.client_id      = client_id
        self.net            = net
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.test_loader    = test_loader
        self.local_epochs   = local_epochs
        self.lr             = lr
        self.device         = device
        self.optimizer_type = optimizer_type
        self.momentum       = momentum
        self.weight_decay   = weight_decay

    def _load_c_local(self):
        """Tải c_i từ disk. Nếu chưa có, khởi tạo bằng zeros."""
        state_file = SCAFFOLD_STATE_DIR / f"c_local_{self.client_id}.pkl"
        if state_file.exists():
            try:
                with open(state_file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return [torch.zeros_like(p.data.cpu()) for p in self.net.parameters()]

    def _load_c_global(self):
        """Tải c_global do Server Strategy ghi ra disk."""
        state_file = SCAFFOLD_STATE_DIR / "c_global.pkl"
        if state_file.exists():
            try:
                with open(state_file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return [torch.zeros_like(p.data.cpu()) for p in self.net.parameters()]

    def _save_c_local(self, c_local_new):
        """Lưu c_i mới vào disk."""
        state_file = SCAFFOLD_STATE_DIR / f"c_local_{self.client_id}.pkl"
        with open(state_file, "wb") as f:
            pickle.dump(c_local_new, f)

    def _save_delta_c(self, delta_c):
        """Lưu delta_c của client để Server đọc và tổng hợp."""
        state_file = SCAFFOLD_STATE_DIR / f"delta_c_{self.client_id}.pkl"
        with open(state_file, "wb") as f:
            pickle.dump(delta_c, f)

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)

        current_lr = config.get("lr", self.lr)
        optimizer = _make_optimizer(
            self.net, self.optimizer_type, current_lr, self.momentum, self.weight_decay
        )

        c_global = self._load_c_global()
        c_local  = self._load_c_local()

        result = train_scaffold(
            model=self.net,
            train_loader=self.train_loader,
            optimizer=optimizer,
            epochs=self.local_epochs,
            device=self.device,
            c_global=c_global,
            c_local=c_local,
            lr=current_lr,
        )

        # Lưu trạng thái mới
        self._save_c_local(result["c_local_new"])
        self._save_delta_c(result["delta_c"])

        fit_metrics = {
            "client_id": self.client_id,
            "tau": float(result["tau"]),
            "num_samples": float(result["num_samples"]),
        }
        return get_parameters(self.net), len(self.train_loader.dataset), fit_metrics

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, eval_metrics = evaluate(self.net, self.test_loader, self.device)
        return loss, len(self.test_loader.dataset), {
            "accuracy":  float(eval_metrics["accuracy"]),
            "precision": float(eval_metrics["precision"]),
            "recall":    float(eval_metrics["recall"]),
            "f1_score":  float(eval_metrics["f1_score"]),
        }


def make_scaffold_client_fn(train_dataset, val_dataset, test_dataset,
                            partition, local_epochs, lr, device,
                            num_classes=11, batch_size=256,
                            optimizer_type="sgd", momentum=0.9,
                            weight_decay=1e-4, model_type="570k"):
    """
    Factory function trả về hàm client_fn cho SCAFFOLD trong Flower Simulation.
    """
    def client_fn(context: Context) -> fl.client.Client:
        client_id = int(context.node_config["partition-id"])

        train_subset = Subset(train_dataset, partition[client_id])
        train_loader = DataLoader(train_subset, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=0)

        in_channels = train_dataset[0][0].shape[0]
        if model_type == "570k":
            net = CNN570K(in_channels=in_channels, num_classes=num_classes).to(device)
        else:
            net = SimpleCNN(in_channels=in_channels, num_classes=num_classes).to(device)

        return ScaffoldFlowerClient(
            client_id=client_id,
            net=net,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            local_epochs=local_epochs,
            lr=lr,
            device=device,
            optimizer_type=optimizer_type,
            momentum=momentum,
            weight_decay=weight_decay,
        ).to_client()

    return client_fn
