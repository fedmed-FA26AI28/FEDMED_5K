import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np
import cloudpickle

import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from algorithms.coverage import coverage_weights, logit_adjustment
from datasets.partition import (
    dirichlet_partition,
    stratified_holdout_indices,
    stratified_subsample_indices,
    stratified_validation_partition,
)
from datasets.medmnist_code import build_transform, fit_train_normalization, get_bloodmnist_dataset
from datasets.sampling import balanced_loader
from models.cnn import build_model, get_parameters, set_parameters
from monitoring.research_validation import evaluate_validation
from experiments.run_simulation import run_one
from server.server import get_strategy
from experiments.run_jetson_client import JetsonFlowerClient
from client.client import make_client_fn


class LabelDataset(Dataset):
    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.zeros(3, 28, 28), self.labels[index]


class ResearchFlowTests(unittest.TestCase):
    def test_ray_client_factory_contains_indices_not_raw_images(self):
        factory = make_client_fn([[0, 1], [2, 3]], [[0], [1]],
                                 local_epochs=1, lr=0.001,
                                 device=torch.device("cpu"))
        self.assertLess(len(cloudpickle.dumps(factory)), 100_000)

    def test_physical_client_monitors_validation_without_test_loader(self):
        dataset = TensorDataset(torch.randn(8, 3, 28, 28), torch.arange(8) % 8)
        loader = DataLoader(dataset, batch_size=4)
        client = JetsonFlowerClient(build_model("tiny_cnn"), loader, loader,
                                    local_epochs=1, lr=0.001,
                                    device=torch.device("cpu"))
        self.assertFalse(hasattr(client, "test_loader"))
        weights, count, _ = client.fit(client.get_parameters({}), {"server_round": 1})
        _, evaluated, _ = client.evaluate(weights, {})
        self.assertEqual(8, count)
        self.assertEqual(8, evaluated)

    def test_final_run_requires_locked_development_spec(self):
        with self.assertRaises(ValueError):
            run_one(2, 0.3, "fedavg", torch.device("cpu"), final_test=True)

    def test_fedavg_uses_parameters_not_client_accuracy(self):
        strategy = get_strategy("fedavg", 2, ndarrays_to_parameters([np.array([0.0])]))
        fit = []
        for count, weight, accuracy in ((2, 3.0, 1.0), (8, 7.0, 0.0)):
            result = FitRes(Status(Code.OK, "ok"),
                            ndarrays_to_parameters([np.array([weight])]), count,
                            {"train_accuracy": accuracy})
            fit.append((SimpleNamespace(cid=str(count)), result))
        parameters, _ = strategy.aggregate_fit(1, fit, [])
        self.assertAlmostEqual(6.2, float(parameters_to_ndarrays(parameters)[0][0]))

    def test_exact_global_validation_uses_only_supplied_indices(self):
        dataset = TensorDataset(torch.randn(16, 3, 28, 28),
                                torch.arange(16) % 8)
        metrics = evaluate_validation(get_parameters(build_model("tiny_cnn")),
                                      "tiny_cnn", dataset, list(range(8)), batch_size=4)
        self.assertEqual("val_monitor", metrics["split"])
        self.assertEqual(8, metrics["num_samples"])
        self.assertIn("f1_macro", metrics)

    def test_64_loader_uses_native_medmnist_plus_size(self):
        with mock.patch("medmnist.BloodMNIST") as blood:
            get_bloodmnist_dataset("train", download=False, size=64)
        self.assertEqual(64, blood.call_args.kwargs["size"])
        self.assertEqual("train", blood.call_args.kwargs["split"])

    def test_validation_transform_is_deterministic_and_train_only(self):
        with self.assertRaises(ValueError):
            build_transform("val", augment=True)
        transforms = build_transform("val").transforms
        self.assertEqual(2, len(transforms))
        self.assertGreater(len(build_transform("train", augment=True).transforms), 2)

    def test_normalization_uses_only_selected_train_images(self):
        dataset = mock.Mock(split="train")
        dataset.imgs = np.stack([np.zeros((2, 2, 3), dtype=np.uint8),
                                 np.full((2, 2, 3), 255, dtype=np.uint8)])
        mean, _ = fit_train_normalization(dataset, [0])
        self.assertEqual((0.0, 0.0, 0.0), mean)
        dataset.split = "test"
        with self.assertRaises(ValueError):
            fit_train_normalization(dataset, [0])

    def test_matched_cnn_and_mobile_head_round_trip(self):
        for name in ("tiny_cnn", "tiny_cnn_gn", "mobilenet_v3_small"):
            model = build_model(name)
            restored = build_model(name)
            set_parameters(restored, get_parameters(model))
            self.assertEqual(8, restored.fc.out_features)
            with torch.no_grad():
                self.assertEqual((2, 8), tuple(restored.eval()(torch.zeros(2, 3, 64, 64)).shape))

    def test_groupnorm_tiny_cnn_has_no_client_running_statistics(self):
        model = build_model("tiny_cnn_gn")
        self.assertFalse(any(isinstance(layer, torch.nn.BatchNorm2d)
                             for layer in model.modules()))
        self.assertEqual(2, sum(isinstance(layer, torch.nn.GroupNorm)
                                for layer in model.modules()))
        self.assertFalse(any("running_mean" in name or "running_var" in name
                             or "num_batches_tracked" in name
                             for name in model.state_dict()))

    def test_balanced_sampler_keeps_local_step_budget(self):
        dataset = LabelDataset([0] * 9 + [1])
        loader = DataLoader(Subset(dataset, list(range(10))), batch_size=3)
        balanced = balanced_loader(loader, torch.tensor([9.0, 1.0]), seed=42)
        self.assertEqual(10, len(list(balanced.sampler)))
        self.assertEqual(len(loader), len(balanced))

    def test_5k_budget_is_applied_before_client_partition(self):
        dataset = LabelDataset([0] * 3000 + [1] * 3000)
        selected = stratified_subsample_indices(dataset, 5000, seed=42)
        partitions = dirichlet_partition(
            dataset, 10, 0.3, seed=42, eligible_indices=selected
        )
        flattened = [index for part in partitions for index in part]
        self.assertEqual(5000, len(flattened))
        self.assertEqual(set(selected), set(flattened))

    def test_calibration_data_is_disjoint_from_client_validation(self):
        dataset = LabelDataset([0] * 20 + [1] * 20)
        monitor, calibration = stratified_holdout_indices(dataset, 0.5, seed=4)
        client_parts = stratified_validation_partition(
            dataset, 5, seed=4, eligible_indices=monitor
        )
        used_by_clients = {index for part in client_parts for index in part}
        self.assertFalse(used_by_clients & set(calibration))
        self.assertEqual(set(monitor), used_by_clients)

    def test_coverage_weight_is_largest_for_missing_class(self):
        counts = torch.tensor([0.0, 8.0, 32.0, 320.0])
        weights = coverage_weights(counts, 32.0)
        self.assertTrue(torch.all(weights[:-1] > weights[1:]))
        self.assertTrue(torch.isfinite(logit_adjustment(counts)).all())


if __name__ == "__main__":
    unittest.main()
