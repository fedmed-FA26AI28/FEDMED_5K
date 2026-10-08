"""Client-local balanced sampling; samples never leave the client."""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets.labels import dataset_labels


def balanced_loader(loader: DataLoader, class_counts: torch.Tensor, seed: int):
    """Draw exactly N local examples per epoch, with inverse-frequency weights."""
    dataset = loader.dataset
    labels = dataset_labels(dataset)
    weights = torch.as_tensor(
        [1.0 / float(class_counts[label]) for label in labels], dtype=torch.double
    )
    generator = torch.Generator().manual_seed(int(seed))
    sampler = WeightedRandomSampler(weights, len(dataset), replacement=True, generator=generator)
    return DataLoader(dataset, batch_size=loader.batch_size, sampler=sampler,
                      num_workers=0, drop_last=loader.drop_last)
