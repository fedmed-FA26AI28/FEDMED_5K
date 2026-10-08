"""Regression tests for the optional vacant-class experiment."""

import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from algorithms.vacant_distillation import (
    frozen_global_teacher,
    vacant_class_distillation_loss,
)
from client.train import train
from experiments.run_jetson_client import JetsonFlowerClient
from models.cnn import build_model
from server.server import get_strategy


class VacantDistillationTests(unittest.TestCase):
    def test_single_vacant_class_has_nonzero_student_only_gradient(self):
        student = torch.tensor([[0.0, -2.0, 0.0]], requires_grad=True)
        teacher = torch.tensor([[0.0, 2.0, 0.0]], requires_grad=True)
        loss = vacant_class_distillation_loss(
            student, teacher, torch.tensor([4, 0, 4]), temperature=2.0,
        )
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(float(student.grad.abs().sum()), 0.0)
        self.assertIsNone(teacher.grad)

    def test_no_vacancy_is_exact_noop(self):
        student = torch.randn(2, 3, requires_grad=True)
        loss = vacant_class_distillation_loss(student, torch.randn(2, 3), torch.ones(3))
        self.assertEqual(0.0, float(loss.detach()))
        loss.backward()
        self.assertEqual(0.0, float(student.grad.abs().sum()))

    def test_teacher_is_frozen_and_disabled_during_warmup(self):
        model = torch.nn.Linear(4, 3)
        counts = torch.tensor([8, 0, 0])
        self.assertIsNone(frozen_global_teacher(model, counts, 0.1, 1))
        self.assertIsNone(frozen_global_teacher(model, torch.ones(3), 0.1, 2))
        teacher = frozen_global_teacher(model, counts, 0.1, 2)
        self.assertFalse(teacher.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in teacher.parameters()))
        original = teacher.weight.detach().clone()
        with torch.no_grad():
            model.weight.add_(1.0)
        self.assertTrue(torch.equal(teacher.weight, original))

    def test_fedavg_strategy_remains_available(self):
        from flwr.common import ndarrays_to_parameters
        import numpy as np
        parameters = ndarrays_to_parameters([np.array([0.0], dtype=np.float32)])
        for name in ("vacant_distill", "coverage_distill"):
            self.assertIsNotNone(get_strategy(name, 2, parameters))

    def test_local_training_uses_teacher_without_updating_it(self):
        torch.manual_seed(17)
        loader = DataLoader(
            TensorDataset(torch.randn(8, 4), torch.zeros(8, dtype=torch.long)),
            batch_size=4,
        )
        model = torch.nn.Linear(4, 3)
        teacher = frozen_global_teacher(model, torch.tensor([8, 0, 0]), 0.1, 2)
        teacher_weight = teacher.weight.detach().clone()
        metrics = train(
            model, loader, torch.optim.SGD(model.parameters(), lr=0.1),
            epochs=1, device=torch.device("cpu"),
            class_counts=torch.tensor([8, 0, 0]),
            global_teacher=teacher, distill_mu=0.1,
        )
        self.assertEqual(8, metrics["num_samples"])
        self.assertGreaterEqual(metrics["final_distill_loss"], 0.0)
        self.assertTrue(torch.equal(teacher.weight, teacher_weight))

    def test_real_client_path_keeps_sample_count_and_uploads_only_model(self):
        torch.manual_seed(19)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        loader = DataLoader(TensorDataset(torch.randn(8, 3, 28, 28), labels), batch_size=4)
        client = JetsonFlowerClient(
            build_model("tiny_cnn"), loader, loader,
            local_epochs=1, lr=0.001, device=torch.device("cpu"),
        )
        parameters, count, metrics = client.fit(client.get_parameters({}), {
            "server_round": 2, "distill_mu": 0.1,
            "distill_max_count": 0, "distill_warmup_rounds": 1,
            "logit_tau": 1.0, "head_mu": 0.01,
        })
        self.assertEqual(8, count)
        self.assertGreaterEqual(metrics["distill_loss"], 0.0)
        self.assertEqual(len(client.get_parameters({})), len(parameters))


if __name__ == "__main__":
    unittest.main()
