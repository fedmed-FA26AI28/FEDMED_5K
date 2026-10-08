"""Exact global-model metrics on the validation monitor subset only."""

import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Subset

from models.cnn import build_model, set_parameters
from monitoring.reliability import collect_logits, probability_metrics


def evaluate_validation(parameters, model_name, val_dataset, indices, num_classes=8,
                        batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, num_classes=num_classes).to(device)
    set_parameters(model, parameters)
    loader = DataLoader(Subset(val_dataset, indices), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    logits, targets = collect_logits(model, loader, device)
    predictions = logits.argmax(dim=1)
    truth, predicted = targets.tolist(), predictions.tolist()
    _, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, labels=list(range(num_classes)), average=None,
        zero_division=0,
    )
    return {
        "split": "val_monitor", "num_samples": len(indices),
        "accuracy": float(predictions.eq(targets).float().mean().item()),
        "f1_macro": float(f1.mean()),
        "worst_class_recall": float(recall.min()),
        "per_class_f1": f1.tolist(), "per_class_recall": recall.tolist(),
        "confusion_matrix": confusion_matrix(
            truth, predicted, labels=list(range(num_classes))
        ).tolist(),
        "calibration": probability_metrics(logits, targets),
    }
