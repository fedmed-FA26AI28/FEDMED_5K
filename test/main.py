"""Read published split counts without loading held-out test images."""

from medmnist import INFO

print(f"Official test split size: {INFO['bloodmnist']['n_samples']['test']}")
