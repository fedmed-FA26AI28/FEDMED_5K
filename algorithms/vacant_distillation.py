"""Client-local, FedVLS-inspired preservation of vacant-class predictions.

This is a lightweight experimental component, not a reproduction of FedVLS:
it does not implement that paper's logit-suppression objective.  The teacher is
the global model received at the start of the round and is never uploaded.
"""

import copy
import math

import torch
import torch.nn.functional as F


def frozen_global_teacher(model, class_counts, strength, server_round,
                          warmup_rounds=1, max_count=0):
    """Create a teacher only when the opt-in objective can actually be active."""
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("distill_mu must be finite and non-negative")
    if warmup_rounds < 0 or max_count < 0:
        raise ValueError("distillation warmup and max_count must be non-negative")
    if strength == 0 or server_round <= warmup_rounds:
        return None
    if class_counts is None or not bool(torch.any(class_counts <= max_count)):
        return None
    teacher = copy.deepcopy(model).eval()
    teacher.requires_grad_(False)
    return teacher


def vacant_class_distillation_loss(student_logits, teacher_logits, class_counts,
                                   temperature=2.0, max_count=0):
    """KL over low-coverage classes plus one aggregated 'other classes' bin.

    The extra bin keeps probability mass meaningful even when only one class is
    vacant; KL over a one-class masked softmax would always be zero.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("distill_temperature must be finite and positive")
    if max_count < 0:
        raise ValueError("distill_max_count must be non-negative")
    if student_logits.ndim != 2 or student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical [batch, class] shapes")
    if class_counts.ndim != 1 or class_counts.numel() != student_logits.shape[1]:
        raise ValueError("class counts must contain one value per logit class")

    selected = (class_counts.to(student_logits.device) <= max_count)
    if not bool(selected.any()):
        return student_logits.sum() * 0.0

    student_log = F.log_softmax(student_logits / temperature, dim=1)
    teacher_prob = F.softmax(teacher_logits.detach() / temperature, dim=1)
    if not bool(selected.all()):
        student_log = torch.cat(
            (student_log[:, selected],
             torch.logsumexp(student_log[:, ~selected], dim=1, keepdim=True)),
            dim=1,
        )
        teacher_prob = torch.cat(
            (teacher_prob[:, selected],
             teacher_prob[:, ~selected].sum(dim=1, keepdim=True)),
            dim=1,
        )
    return F.kl_div(student_log, teacher_prob, reduction="batchmean") * temperature**2
