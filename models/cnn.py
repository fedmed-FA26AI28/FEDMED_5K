import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np

    
class ConvBlock(nn.Module):
    """
    Khối Convolutional cơ bản:
    Conv2d (stride=1, padding=1) -> BatchNorm2d -> ReLU
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class SimpleCNN(nn.Module):
    """
    Kiến trúc mạng CNN tối ưu cho bài toán phân loại ảnh y tế (MedMNIST - BloodMNIST 28x28x3):
    
    Input (3, 28, 28)
           ↓
       Conv Block 1 (3 -> 16, 28x28)
           ↓
       Conv Block 2 (16 -> 32, 28x28)
           ↓
       Pooling (MaxPool2d 2x2 -> 32, 14x14)
           ↓
       Global Average Pooling (GAP -> 32, 1x1)
           ↓
       Linear (32 -> num_classes)
           ↓
       Output (num_classes = 8)
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 8):
        super(SimpleCNN, self).__init__()
        # 2 khối Conv liên tiếp
        self.conv_block1 = ConvBlock(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv_block2 = ConvBlock(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Lớp Pooling giảm kích thước không gian (28x28 -> 14x14)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global Average Pooling (nén 14x14 về 1x1 cho mỗi channel)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # 1 tầng phân loại tuyến tính duy nhất
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)  # [B, 16, 28, 28]
        x = self.conv_block2(x)  # [B, 32, 28, 28]
        x = self.pool(x)         # [B, 32, 14, 14]
        x = self.gap(x)          # [B, 32, 1, 1]
        x = torch.flatten(x, 1)  # [B, 32]
        x = self.fc(x)           # [B, num_classes]
        return x


class ResearchCNN(nn.Module):
    """The same 32/64-channel tiny CNN used in the full-train project."""
    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.conv_block1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.conv_block2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(self.conv_block2(self.conv_block1(x)))
        return self.fc(torch.flatten(self.global_avg_pool(x), 1))


class GroupNormResearchCNN(ResearchCNN):
    """Same research CNN, with per-example normalization and no running BN state."""

    def __init__(self, num_classes: int = 8):
        super().__init__(num_classes=num_classes)
        self.conv_block1[1] = nn.GroupNorm(8, 32)
        self.conv_block2[1] = nn.GroupNorm(8, 64)


class MobileNetSmall(nn.Module):
    """Untrained MobileNetV3-Small with an exposed linear FL classifier head."""
    def __init__(self, num_classes: int = 8):
        super().__init__()
        from torchvision.models import mobilenet_v3_small
        self.backbone = mobilenet_v3_small(weights=None)
        features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Identity()
        self.fc = nn.Linear(features, num_classes)

    def forward(self, x):
        return self.fc(self.backbone(x))


def build_model(name: str = "legacy", num_classes: int = 8):
    """Build a shared architecture for the full and 5K training budgets."""
    factories = {
        "legacy": lambda: SimpleCNN(num_classes=num_classes),
        "tiny_cnn": lambda: ResearchCNN(num_classes=num_classes),
        "tiny_cnn_gn": lambda: GroupNormResearchCNN(num_classes=num_classes),
        "mobilenet_v3_small": lambda: MobileNetSmall(num_classes=num_classes),
    }
    if name not in factories:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(factories)}")
    return factories[name]()


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Trích xuất trọng số mô hình PyTorch thành danh sách các mảng NumPy."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Nạp danh sách trọng số NumPy vào mô hình PyTorch."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


if __name__ == "__main__":
    # Chạy kiểm thử kiến trúc mô hình
    model = SimpleCNN(in_channels=3, num_classes=8)
    dummy_input = torch.randn(4, 3, 28, 28)
    output = model(dummy_input)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=== MODEL TEST: SimpleCNN ===")
    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total trainable parameters: {total_params:,}")
    assert output.shape == (4, 8), "Error: Output shape is not (4, 8)!"
    print("Test passed successfully!")
