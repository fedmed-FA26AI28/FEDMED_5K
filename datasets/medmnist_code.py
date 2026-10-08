"""Load and preprocess BloodMNIST dataset splits."""

import torchvision.transforms as transforms
import numpy as np
from functools import lru_cache
from medmnist import INFO
from torch.utils.data import DataLoader


def build_transform(split: str, augment: bool = False, normalization=None):
    """Only a training split may receive random image transformations."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")
    if augment and split != "train":
        raise ValueError("augmentation is permitted only on train")
    mean, std = normalization or ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    operations = []
    if augment:
        operations += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
        ]
    return transforms.Compose(operations + [
        transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)
    ])


def fit_train_normalization(dataset, eligible_indices):
    """Fit RGB moments on permitted raw *training* images, never val/test."""
    if getattr(dataset, "split", None) != "train":
        raise ValueError("normalization statistics require the train split")
    indices = np.asarray(eligible_indices, dtype=np.int64)
    if not len(indices):
        raise ValueError("training index set cannot be empty")
    images = dataset.imgs
    sums = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    pixels = 0
    for start in range(0, len(indices), 256):
        batch = np.asarray(images[indices[start:start + 256]], dtype=np.float64) / 255.0
        sums += batch.sum(axis=(0, 1, 2))
        squares += np.square(batch).sum(axis=(0, 1, 2))
        pixels += batch.shape[0] * batch.shape[1] * batch.shape[2]
    mean = sums / pixels
    std = np.sqrt(np.maximum(squares / pixels - mean ** 2, 1e-12))
    return tuple(mean.tolist()), tuple(std.tolist())


def get_bloodmnist_dataset(split: str, download: bool = True, *, size: int = 28,
                           augment: bool = False, normalization=None):
    """Load exactly one split so FL never loads test data before the end."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")
    if size not in {28, 64}:
        raise ValueError("supported BloodMNIST sizes are 28 and 64")

    info = INFO["bloodmnist"]
    num_classes = len(info["label"])
    import medmnist as _medmnist

    data_class = getattr(_medmnist, info["python_class"])
    transform = build_transform(split, augment=augment, normalization=normalization)
    kwargs = {"split": split, "transform": transform, "download": download}
    if size != 28:
        kwargs["size"] = size
    return (
        data_class(**kwargs),
        num_classes,
    )


@lru_cache(maxsize=8)
def load_simulation_datasets(size, augment, normalization):
    """Worker-local public data; only indices enter the Ray client factory."""
    train, _ = get_bloodmnist_dataset(
        "train", download=False, size=size, augment=augment,
        normalization=normalization,
    )
    val, _ = get_bloodmnist_dataset(
        "val", download=False, size=size, normalization=normalization,
    )
    return train, val


def get_bloodmnist_datasets(batch_size: int = 32, download: bool = True):
    """Return all dataset splits for the centralized baseline."""
    train_dataset, num_classes = get_bloodmnist_dataset("train", download)
    val_dataset, _ = get_bloodmnist_dataset("val", download)
    test_dataset, _ = get_bloodmnist_dataset("test", download)
    return train_dataset, val_dataset, test_dataset, num_classes


def get_bloodmnist_dataloaders(batch_size: int = 32, download: bool = True):
    """Return train/validation/test loaders for centralized training."""
    train_dataset, val_dataset, test_dataset, num_classes = (
        get_bloodmnist_datasets(download=download)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, val_loader, test_loader, num_classes
