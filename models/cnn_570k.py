"""
Mô hình 4-layer CNN chuẩn của bài báo FedSLD (Luo & Wu, IEEE ISBI 2022 / McMahan et al. 2017).
Quy mô tham số: ~570,000 params (571,707 cho 11 class OrganMNIST; 570,705 cho 9 class PathMNIST).
Kiến trúc:
  - Conv1: 5x5, 32 channels, ReLU, MaxPool(2) -> 12x12
  - Conv2: 5x5, 64 channels, ReLU, MaxPool(2) -> 4x4
  - FC1  : (64 * 4 * 4 = 1024) -> 500 units, ReLU
  - FC2  : 500 -> num_classes
"""

from collections import OrderedDict
from typing import List
import numpy as np
import torch
import torch.nn as nn


class CNN570K(nn.Module):
    """
    Classic 4-layer CNN tái hiện nguyên bản kiến trúc trong FedSLD (ISBI 2022).
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 11):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, 32, kernel_size=5)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2', nn.Conv2d(32, 64, kernel_size=5)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 500)),
            ('relu3', nn.ReLU()),
            ('fc2', nn.Linear(500, num_classes)),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.classifier(x)
        return x


class CNN570K_BN(nn.Module):
    """
    Classic 4-layer CNN (~570K params) tích hợp các lớp BatchNorm2d
    cho phép triển khai FedBN (giữ các tham số và running stats của BN cục bộ).
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 11):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, 32, kernel_size=5)),
            ('bn1',   nn.BatchNorm2d(32)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2', nn.Conv2d(32, 64, kernel_size=5)),
            ('bn2',   nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 500)),
            ('relu3', nn.ReLU()),
            ('fc2', nn.Linear(500, num_classes)),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.classifier(x)
        return x


def get_parameters_fedbn(model: nn.Module) -> List[np.ndarray]:
    """
    FedBN: Chỉ trích xuất các tham số KHÔNG phải BatchNorm.
    BN params (weight, bias, running_mean, running_var, num_batches_tracked)
    sẽ được giữ lại cục bộ tại mỗi client.
    """
    return [
        val.cpu().numpy()
        for name, val in model.state_dict().items()
        if "bn" not in name
    ]


def set_parameters_fedbn(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """
    FedBN: Chỉ nạp các tham số KHÔNG phải BatchNorm vào model.
    Các tham số BN của client sẽ không bị ghi đè bởi Server.
    """
    non_bn_keys = [
        name for name in model.state_dict().keys()
        if "bn" not in name
    ]
    state_dict = model.state_dict()
    state_dict.update({
        k: torch.tensor(v)
        for k, v in zip(non_bn_keys, parameters)
    })
    model.load_state_dict(state_dict, strict=True)


def get_parameters(net: nn.Module) -> List[np.ndarray]:
    """Trích xuất trọng số của mô hình dưới dạng danh sách NumPy array."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net: nn.Module, parameters: List[np.ndarray]) -> None:
    """Nạp danh sách NumPy array từ server vào mô hình."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

