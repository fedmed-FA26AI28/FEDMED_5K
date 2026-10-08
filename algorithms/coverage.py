"""Client-private class-coverage-aware local objective components."""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from datasets.labels import dataset_labels


def count_client_classes(dataset, num_classes: int) -> torch.Tensor:
    labels = dataset_labels(dataset)
    if np.any((labels < 0) | (labels >= num_classes)):
        raise ValueError("training label outside configured class range")
    return torch.bincount(torch.as_tensor(labels), minlength=num_classes).float()


def logit_adjustment(class_counts, tau=1.0, smoothing=1.0):
    if tau < 0 or smoothing <= 0:
        raise ValueError("tau must be non-negative and smoothing must be positive")
    return tau * torch.log(class_counts.float() + smoothing)


def snapshot_classifier_head(model: nn.Module) -> Dict[str, torch.Tensor]:
    if not hasattr(model, "fc") or not isinstance(model.fc, nn.Linear):
        raise ValueError("Coverage-aware training requires model.fc to be nn.Linear")
    return {
        "weight": model.fc.weight.detach().clone(),
        "bias": model.fc.bias.detach().clone(),
    }


def coverage_weights(class_counts, kappa):
    if kappa <= 0:
        raise ValueError("coverage_kappa must be greater than zero")
    return kappa / (class_counts.float() + kappa)


def coverage_head_penalty(model, global_head, class_counts, head_mu, kappa):
    weights = coverage_weights(class_counts, kappa).to(model.fc.weight.device)
    weight_distance = (model.fc.weight - global_head["weight"]).pow(2).sum(dim=1)
    bias_distance = (model.fc.bias - global_head["bias"]).pow(2)
    return 0.5 * head_mu * torch.sum(weights * (weight_distance + bias_distance))
