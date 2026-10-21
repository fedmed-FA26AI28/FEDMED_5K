"""
Vòng lặp huấn luyện cục bộ cho FedNova (Federated Normalized Averaging - Li et al., NeurIPS 2020).
Theo dõi và trả về số bước cập nhật cục bộ (tau) để Server chuẩn hóa gradient.
"""

import time
import torch
import torch.nn as nn


def train_fednova(model, train_loader, optimizer, epochs, device,
                  proximal_mu: float = 0.0, global_params: list = None):
    """
    Huấn luyện cục bộ cho FedNova:
      - Thực hiện SGD qua các batch trong train_loader
      - Đếm chính xác số bước mini-batch SGD: tau
      - Trả về metrics bao gồm 'tau', 'num_samples', 'training_time'
    """
    criterion = nn.CrossEntropyLoss()
    model.train()

    start_time = time.time()
    tau = 0
    total = 0
    running_loss = 0.0
    correct = 0

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.squeeze(-1).long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Hỗ trợ thêm proximal term nếu kết hợp proximal regularization
            if proximal_mu > 0.0 and global_params is not None:
                prox_term = 0.0
                for w, w_t in zip(model.parameters(), global_params):
                    prox_term += ((w - w_t) ** 2).sum()
                loss += (proximal_mu / 2.0) * prox_term

            loss.backward()
            optimizer.step()
            tau += 1

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    training_time = time.time() - start_time
    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0

    current_lr = optimizer.param_groups[0]['lr']
    print(f"   [FedNova Client] Epochs: {epochs} | Steps (tau): {tau} (LR: {current_lr:.6f}) | Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

    return {
        "tau": tau,
        "training_time": training_time,
        "num_samples": len(train_loader.dataset),
        "final_loss": epoch_loss,
        "final_accuracy": epoch_acc,
    }
