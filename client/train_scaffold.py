"""
Vòng lặp huấn luyện cục bộ cho SCAFFOLD (Karimireddy et al., ICML 2020) — Option II.
Sử dụng control variates để hiệu chỉnh hiện tượng client drift trong môi trường Non-IID.
"""

import time
import torch
import torch.nn as nn


def train_scaffold(model, train_loader, optimizer, epochs, device,
                   c_global: list, c_local: list, lr: float):
    """
    Huấn luyện cục bộ theo SCAFFOLD (Option II):
      - Gradient tại mỗi bước được hiệu chỉnh: grad = grad - c_i + c
      - Cập nhật c_i mới sau vòng lặp:
          c_i_new = c_i - c + (1 / (tau * lr)) * (w_0 - w_tau)
          delta_c = c_i_new - c_i = (1 / (tau * lr)) * (w_0 - w_tau) - c
    """
    criterion = nn.CrossEntropyLoss()
    model.train()

    # 1. Lưu bản sao w_0 (trọng số trước khi huấn luyện)
    w_0 = [p.detach().clone().to("cpu") for p in model.parameters()]

    # Chuyển control variates sang device để tính toán nhanh
    c_g_dev = [cg.to(device) for cg in c_global]
    c_l_dev = [cl.to(device) for cl in c_local]

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
            loss.backward()
            optimizer.step()

            # Chuẩn NVIDIA NVFlare / NIID-Bench:
            # Hiệu chỉnh control variate trực tiếp lên tham số mô hình SAU KHI optimizer.step().
            # Điều này giúp momentum buffer (0.9) KHÔNG bị nhiễm bẩn bởi control variate,
            # triệt tiêu hoàn toàn hiện tượng khuếch đại cấp số nhân dẫn đến NaN!
            with torch.no_grad():
                for p, cg, cl in zip(model.parameters(), c_g_dev, c_l_dev):
                    p.data.add_(cg - cl, alpha=-lr)
            tau += 1

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    training_time = time.time() - start_time
    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0

    # 2. Cập nhật control variate cục bộ (Option II)
    # c_i_new = c_i - c + (1 / (tau * lr)) * (w_0 - w_final)
    # delta_c = c_i_new - c_i
    w_final = [p.detach().clone().to("cpu") for p in model.parameters()]
    c_local_new = []
    delta_c = []

    scale = 1.0 / (tau * lr) if (tau > 0 and lr > 0) else 1.0

    for cl, cg, w0_p, wf_p in zip(c_local, c_global, w_0, w_final):
        # Tính theo cpu để tiết kiệm VRAM
        diff_w = (w0_p - wf_p) * scale
        c_new = cl.to("cpu") - cg.to("cpu") + diff_w
        dc = c_new - cl.to("cpu")
        c_local_new.append(c_new)
        delta_c.append(dc)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"   [SCAFFOLD Client] Epochs: {epochs} | Steps (tau): {tau} (LR: {current_lr:.6f}) | Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

    return {
        "c_local_new": c_local_new,
        "delta_c": delta_c,
        "tau": tau,
        "training_time": training_time,
        "num_samples": len(train_loader.dataset),
        "final_loss": epoch_loss,
        "final_accuracy": epoch_acc,
    }
