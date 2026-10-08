"""Flower virtual clients used by the single-machine FL simulation."""

import flwr as fl
import numpy as np
import torch
from flwr.common import Context
from torch.utils.data import DataLoader, Subset

from algorithms.coverage import (
    count_client_classes,
    coverage_weights,
    snapshot_classifier_head,
)
from algorithms.vacant_distillation import frozen_global_teacher
from client.evaluate import evaluate
from client.train import train
from datasets.sampling import balanced_loader
from datasets.medmnist_code import load_simulation_datasets
from models.cnn import build_model, get_parameters, set_parameters


def make_client_fn(
    train_partition,
    val_partition,
    local_epochs,
    lr,
    device,
    num_classes=8,
    batch_size=32,
    model_name="legacy",
    seed=42,
    size=28,
    augment=False,
    normalization=None,
):
    """Return the factory Flower uses to create isolated virtual clients."""

    def client_fn(context: Context) -> fl.client.Client:
        client_id = int(context.node_config["partition-id"])
        train_dataset, val_dataset = load_simulation_datasets(size, augment, normalization)
        client_train = DataLoader(
            Subset(train_dataset, train_partition[client_id]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        client_val = DataLoader(
            Subset(val_dataset, val_partition[client_id]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        net = build_model(model_name, num_classes=num_classes).to(device)
        return FlowerClient(
            net=net,
            client_train=client_train,
            client_val=client_val,
            local_epochs=local_epochs,
            lr=lr,
            device=device,
            client_id=client_id,
            seed=seed,
        ).to_client()

    return client_fn


class FlowerClient(fl.client.NumPyClient):
    """A simulated client that never receives or loads the global test set."""

    def __init__(
        self, net, client_train, client_val, local_epochs, lr, device,
        client_id=0, seed=42,
    ):
        self.net = net
        self.client_train = client_train
        self.client_val = client_val
        self.local_epochs = local_epochs
        self.lr = lr
        self.device = device
        self.client_id = client_id
        self.seed = seed
        self.class_counts = count_client_classes(
            self.client_train.dataset, self.net.fc.out_features
        )

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        """Receive global weights, train locally, and return model weights."""
        round_seed = self.seed + 1009 * int(config.get("server_round", 0)) + self.client_id
        torch.manual_seed(round_seed)
        np.random.seed(round_seed % (2 ** 32 - 1))
        set_parameters(self.net, parameters)
        current_lr = float(config.get("lr", self.lr))
        logit_tau = float(config.get("logit_tau", 0.0))
        prior_smoothing = float(config.get("prior_smoothing", 1.0))
        head_mu = float(config.get("head_mu", 0.0))
        proximal_mu = float(config.get("proximal_mu", 0.0))
        coverage_kappa = float(config.get("coverage_kappa", 32.0))
        global_head = snapshot_classifier_head(self.net) if head_mu > 0 else None
        distill_mu = float(config.get("distill_mu", 0.0))
        distill_max_count = int(config.get("distill_max_count", 0))
        global_teacher = frozen_global_teacher(
            self.net, self.class_counts, distill_mu,
            server_round=int(config.get("server_round", 0)),
            warmup_rounds=int(config.get("distill_warmup_rounds", 1)),
            max_count=distill_max_count,
        )
        optimizer = torch.optim.Adam(self.net.parameters(), lr=current_lr)
        train_loader = self.client_train
        if bool(config.get("balanced_sampling", False)):
            train_loader = balanced_loader(
                self.client_train, self.class_counts,
                seed=self.seed + 1009 * int(config.get("server_round", 0)) + self.client_id,
            )
        metrics = train(
            model=self.net,
            train_loader=train_loader,
            optimizer=optimizer,
            epochs=self.local_epochs,
            device=self.device,
            val_loader=self.client_val,
            class_counts=self.class_counts,
            logit_tau=logit_tau,
            prior_smoothing=prior_smoothing,
            head_mu=head_mu,
            coverage_kappa=coverage_kappa,
            global_head=global_head,
            proximal_mu=proximal_mu,
            global_params=(
                [parameter.detach().clone() for parameter in self.net.parameters()]
                if proximal_mu > 0 else None
            ),
            global_teacher=global_teacher,
            distill_mu=distill_mu,
            distill_temperature=float(config.get("distill_temperature", 2.0)),
            distill_max_count=distill_max_count,
        )
        result_metrics = {
            "train_loss": float(metrics["final_loss"]),
            "train_accuracy": float(metrics["final_accuracy"]),
            "val_loss": float(metrics.get("final_val_loss", 0.0)),
            "val_accuracy": float(metrics.get("final_val_accuracy", 0.0)),
            "num_val_samples": int(len(self.client_val.dataset)),
            "num_classes_present": int((self.class_counts > 0).sum().item()),
            "logit_tau": logit_tau,
            "head_mu": head_mu,
            "distill_loss": float(metrics["final_distill_loss"]),
            "distill_active_classes": int((self.class_counts <= distill_max_count).sum().item()) if global_teacher is not None else 0,
        }
        if head_mu > 0:
            result_metrics["coverage_weight_mean"] = float(
                coverage_weights(self.class_counts, coverage_kappa).mean().item()
            )
        return get_parameters(self.net), len(self.client_train.dataset), result_metrics

    def evaluate(self, parameters, config):
        """Evaluate global weights on this client's validation subset."""
        set_parameters(self.net, parameters)
        loss, metrics = evaluate(self.net, self.client_val, self.device)
        return loss, len(self.client_val.dataset), {
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1_score": float(metrics["f1_score"]),
        }
