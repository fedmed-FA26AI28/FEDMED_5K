"""Read labels without applying image augmentations or consuming RNG state."""

import numpy as np
from torch.utils.data import Subset


def dataset_labels(dataset):
    if isinstance(dataset, Subset):
        return dataset_labels(dataset.dataset)[np.asarray(dataset.indices, dtype=np.int64)]
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels, dtype=np.int64).reshape(-1)
    return np.asarray([int(np.asarray(dataset[i][1]).squeeze())
                       for i in range(len(dataset))], dtype=np.int64)
