"""FL Client - Wrapper Flower cho SimpleCNN."""

import torch
import flwr as fl
from models.cnn import get_parameters, set_parameters, SimpleCNN
from models.cnn_570k import CNN570K
from flwr.common import Context
from client.train import train
from client.evaluate import evaluate
from models.cnn import get_parameters_fedbn, set_parameters_fedbn


def _make_optimizer(net, optimizer_type, lr, momentum=0.0, weight_decay=0.0):
    """Khởi tạo optimizer: SGD (theo FedSLD paper) hoặc Adam (theo BloodMNIST)."""
    if optimizer_type == "sgd":
        return torch.optim.SGD(
            net.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    return torch.optim.Adam(net.parameters(), lr=lr)


def make_client_fn(train_dataset, val_dataset, test_dataset,
                   partition, local_epochs, lr, device,
                   num_classes=8, batch_size=32,
                   optimizer_type="adam", momentum=0.0, weight_decay=0.0,
                   model_type="20k", use_la=False, tau_la=1.0):
    """
    Factory function: trả về hàm client_fn dùng cho Flower Simulation.
    """
    import numpy as np
    from torch.utils.data import DataLoader, Subset

    def client_fn(context: Context) -> fl.client.Client:
        client_id = int(context.node_config["partition-id"])

        train_subset = Subset(train_dataset, partition[client_id])
        train_loader = DataLoader(train_subset, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_dataset,  batch_size=batch_size,
                                  shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=0)

        log_priors = None
        if use_la and hasattr(train_dataset, 'labels'):
            client_labels = train_dataset.labels[partition[client_id]].reshape(-1)
            counts = np.bincount(client_labels, minlength=num_classes).astype(np.float32)
            counts = counts + 1e-4
            priors = counts / counts.sum()
            log_priors = torch.tensor(np.log(priors), dtype=torch.float32)

        in_channels = train_dataset[0][0].shape[0]
        if model_type == "570k":
            net = CNN570K(in_channels=in_channels, num_classes=num_classes).to(device)
        else:
            net = SimpleCNN(in_channels=in_channels, num_classes=num_classes).to(device)

        return FlowerClient(
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
            log_priors=log_priors,
            tau_la=tau_la,
        ).to_client()

    return client_fn


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, train_loader, val_loader,
                 test_loader, local_epochs, lr, device,
                 optimizer_type="adam", momentum=0.0, weight_decay=0.0,
                 log_priors=None, tau_la=1.0):
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
        self.log_priors     = log_priors
        self.tau_la         = tau_la

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        """Nhận trọng số từ Server -> train cục bộ -> gửi về."""
        set_parameters(self.net, parameters)

        current_lr  = config.get("lr", self.lr)
        proximal_mu = float(config.get("proximal_mu", 0.0))
        tau_la      = float(config.get("tau_la", self.tau_la))

        # Lưu bản sao trọng số global đầu round cho FedProx
        global_params = None
        if proximal_mu > 0.0:
            global_params = [p.detach().clone() for p in self.net.parameters()]

        optimizer = _make_optimizer(self.net, self.optimizer_type,
                                    current_lr, self.momentum, self.weight_decay)
        metrics = train(
            model=self.net,
            train_loader=self.train_loader,
            optimizer=optimizer,
            epochs=self.local_epochs,
            device=self.device,
            val_loader=None,
            proximal_mu=proximal_mu,
            global_params=global_params,
            log_priors=self.log_priors,
            tau_la=tau_la,
        )
        fit_metrics = {
            "train_loss": float(metrics.get("final_loss", 0.0)),
            "train_accuracy": float(metrics.get("final_accuracy", 0.0)),
        }
        return get_parameters(self.net), len(self.train_loader.dataset), fit_metrics

    def evaluate(self, parameters, config):
        """Nhận trọng số từ Server -> đánh giá trên test set."""
        set_parameters(self.net, parameters)
        loss, eval_metrics = evaluate(self.net, self.test_loader, self.device)
        return loss, len(self.test_loader.dataset), {
            "accuracy":  float(eval_metrics["accuracy"]),
            "precision": float(eval_metrics["precision"]),
            "recall":    float(eval_metrics["recall"]),
            "f1_score":  float(eval_metrics["f1_score"]),
        }


class FedBNFlowerClient(fl.client.NumPyClient):
    """
    Client cho FedBN: BN params được giữ lại cục bộ,
    chỉ các tham số Conv và FC mới được gửi lên Server.
    """
    def __init__(self, net, train_loader, val_loader,
                 test_loader, local_epochs, lr, device,
                 optimizer_type="adam", momentum=0.0, weight_decay=0.0):
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

    def get_parameters(self, config):
        return get_parameters_fedbn(self.net)   # Bỏ BN params

    def fit(self, parameters, config):
        set_parameters_fedbn(self.net, parameters)  # Không ghi đè BN cục bộ

        current_lr  = config.get("lr", self.lr)
        proximal_mu = float(config.get("proximal_mu", 0.0))

        global_params = None
        if proximal_mu > 0.0:
            global_params = [p.detach().clone() for p in self.net.parameters()]

        optimizer = _make_optimizer(self.net, self.optimizer_type,
                                    current_lr, self.momentum, self.weight_decay)
        metrics = train(
            model=self.net,
            train_loader=self.train_loader,
            optimizer=optimizer,
            epochs=self.local_epochs,
            device=self.device,
            val_loader=None,
            proximal_mu=proximal_mu,
            global_params=global_params,
        )
        fit_metrics = {
            "train_loss": float(metrics.get("final_loss", 0.0)),
            "train_accuracy": float(metrics.get("final_accuracy", 0.0)),
        }
        return get_parameters_fedbn(self.net), len(self.train_loader.dataset), fit_metrics

    def evaluate(self, parameters, config):
        set_parameters_fedbn(self.net, parameters)
        loss, eval_metrics = evaluate(self.net, self.test_loader, self.device)
        return loss, len(self.test_loader.dataset), {
            "accuracy":  float(eval_metrics["accuracy"]),
            "precision": float(eval_metrics["precision"]),
            "recall":    float(eval_metrics["recall"]),
            "f1_score":  float(eval_metrics["f1_score"]),
        }


def make_fedbn_client_fn(train_dataset, val_dataset, test_dataset,
                         partition, local_epochs, lr, device,
                         num_classes=8, batch_size=32,
                         optimizer_type="adam", momentum=0.0, weight_decay=0.0,
                         model_type="20k"):
    """Factory function cho FedBN — tương tự make_client_fn."""
    from torch.utils.data import DataLoader, Subset

    def client_fn(context: Context) -> fl.client.Client:
        client_id    = int(context.node_config["partition-id"])
        train_subset = Subset(train_dataset, partition[client_id])
        train_loader = DataLoader(train_subset, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_dataset,  batch_size=batch_size,
                                  shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=0)
        in_channels = train_dataset[0][0].shape[0]
        if model_type == "570k":
            net = CNN570K(in_channels=in_channels, num_classes=num_classes).to(device)
        else:
            net = SimpleCNN(in_channels=in_channels, num_classes=num_classes).to(device)
        return FedBNFlowerClient(
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