"""Post-hoc calibration and conformal metrics for held-out evaluation."""

import math

import torch
import torch.nn.functional as F


def collect_logits(model, loader, device):
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            logits.append(model(images.to(device)).detach().cpu())
            labels.append(targets.squeeze().long().cpu())
    if not logits:
        raise ValueError("Cannot evaluate an empty dataset")
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(logits, labels):
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_temperature.exp().clamp(0.05, 20.0), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())


def probability_metrics(logits, labels, temperature=1.0, num_bins=15):
    probabilities = torch.softmax(logits / temperature, dim=1)
    confidence, predictions = probabilities.max(dim=1)
    correctness = predictions.eq(labels).float()
    one_hot = F.one_hot(labels, num_classes=logits.shape[1]).float()
    ece = torch.tensor(0.0)
    boundaries = torch.linspace(0.0, 1.0, num_bins + 1)
    for index in range(num_bins):
        mask = (confidence > boundaries[index]) & (confidence <= boundaries[index + 1])
        if mask.any():
            ece += mask.float().mean() * torch.abs(
                correctness[mask].mean() - confidence[mask].mean()
            )
    return {
        "nll": float(F.cross_entropy(logits / temperature, labels).item()),
        "brier": float(((probabilities - one_hot) ** 2).sum(dim=1).mean().item()),
        "ece": float(ece.item()),
    }


def fit_conformal_threshold(logits, labels, alpha=0.1, temperature=1.0):
    probabilities = torch.softmax(logits / temperature, dim=1)
    scores = 1.0 - probabilities[torch.arange(len(labels)), labels]
    rank = math.ceil((len(scores) + 1) * (1.0 - alpha)) - 1
    return float(torch.sort(scores).values[min(max(rank, 0), len(scores) - 1)].item())


def conformal_metrics(logits, labels, threshold, temperature=1.0):
    probabilities = torch.softmax(logits / temperature, dim=1)
    sets = (1.0 - probabilities) <= threshold
    covered = sets[torch.arange(len(labels)), labels]
    classwise = {}
    for class_id in range(logits.shape[1]):
        mask = labels == class_id
        if mask.any():
            classwise[str(class_id)] = float(covered[mask].float().mean().item())
    return {
        "coverage": float(covered.float().mean().item()),
        "average_set_size": float(sets.sum(dim=1).float().mean().item()),
        "empty_set_rate": float((sets.sum(dim=1) == 0).float().mean().item()),
        "worst_class_coverage": float(min(classwise.values())),
        "classwise_coverage": classwise,
    }
