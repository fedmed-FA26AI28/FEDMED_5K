"""Chịu trách nhiệm tải, tiền xử lý (preprocess) và cung cấp data loader cho các dataset MedMNIST (như BloodMNIST)."""

import torch
import torchvision.transforms as transforms
from medmnist import INFO
import os
from pathlib import Path
from torch.utils.data import DataLoader

_DATA_MEDMNIST = Path("data/medmnist")
MEDMNIST_ROOT = str(_DATA_MEDMNIST.resolve()) if _DATA_MEDMNIST.exists() else os.path.expanduser("~/.medmnist")


# def get_bloodmnist_datasets(batch_size: int = 32, download: bool = True) :
#     """
#     Hàm khởi tạo DataLoaders cho bộ dữ liệu BloodMNIST.
    
#     Args:
#         batch_size (int): Kích thước batch.
#         download (bool): Có tự động tải dữ liệu nếu chưa có hay không.
        
#     Returns:
#         train_dataset, val_dataset, test_dataset, num_classes
#     """

#     # khai bao dataset
#     data_flag = 'bloodmnist'
#     info = INFO[data_flag]
#     num_classes = len(info['label'])
#     # Import bên trong hàm để tránh circular import (file tên trùng với package)
#     import medmnist as _medmnist
#     DataClass = getattr(_medmnist, info['python_class'])

#     #Transform data
#     data_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])

#     # Load dataset
#     train_dataset = DataClass(split = 'train', transform = data_transform, download = download)
#     val_dataset = DataClass(split = 'val', transform = data_transform, download = download)
#     test_dataset = DataClass(split = 'test', transform = data_transform, download = download)

#     return train_dataset, val_dataset, test_dataset, num_classes


def get_pathmnist_datasets(download: bool = True):
    """
    PathMNIST: 9-class, 3-channel RGB, 28x28.
    Colorectal cancer histology images.
    Train: 89,996 | Val: 10,004 | Test: 7,180
    """
    import medmnist as _medmnist
    info = INFO['pathmnist']
    DataClass = getattr(_medmnist, info['python_class'])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])
    train_ds = DataClass(split='train', transform=transform, download=download, root=MEDMNIST_ROOT)
    val_ds   = DataClass(split='val',   transform=transform, download=download, root=MEDMNIST_ROOT)
    test_ds  = DataClass(split='test',  transform=transform, download=download, root=MEDMNIST_ROOT)
    return train_ds, val_ds, test_ds, 9


def get_organmnist_datasets(download: bool = True):
    """
    OrganMNIST (Axial): 11-class, 1-channel grayscale → replicate to 3-ch.
    Abdominal CT organ images.
    Train: 34,581 | Val: 6,491 | Test: 17,778
    """
    import medmnist as _medmnist
    info = INFO['organamnist']
    DataClass = getattr(_medmnist, info['python_class'])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    train_ds = DataClass(split='train', transform=transform, download=download, root=MEDMNIST_ROOT)
    val_ds   = DataClass(split='val',   transform=transform, download=download, root=MEDMNIST_ROOT)
    test_ds  = DataClass(split='test',  transform=transform, download=download, root=MEDMNIST_ROOT)
    return train_ds, val_ds, test_ds, 11


# def get_bloodmnist_dataloaders(batch_size: int = 32, download: bool = True):
#     """
#     Trả về DataLoaders phục vụ cho Centralized Baseline (Train trực tiếp toàn bộ dữ liệu).
#     """
#     train_dataset, val_dataset, test_dataset, num_classes = get_bloodmnist_datasets(download = download)
#     train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#     val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
#     test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
#     return train_loader, val_loader, test_loader, num_classes


