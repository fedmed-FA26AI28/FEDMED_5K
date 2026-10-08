"""Verify official train/val indices match at 28 and 64; never open test."""

import json
import numpy as np

from datasets.medmnist_code import get_bloodmnist_dataset


def verify():
    result = {}
    for split in ("train", "val"):
        low, _ = get_bloodmnist_dataset(split, download=True, size=28)
        high, _ = get_bloodmnist_dataset(split, download=True, size=64)
        aligned = len(low) == len(high) and np.array_equal(low.labels, high.labels)
        if not aligned:
            raise AssertionError(f"{split} labels or indices differ across resolutions")
        result[split] = {"samples": len(low), "labels_aligned": True,
                         "low_size": low.size, "high_size": high.size}
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
