"""
Vòng lặp huấn luyện cục bộ cho FedBN + SCAFFOLD + Logit Adjustment (FedBN-SCAFFOLD-LA).
Kết hợp:
  1. FedBN: Các lớp BatchNorm (weight, bias, running stats) được huấn luyện và giữ lại cục bộ.
  2. Logit Adjustment (LA): Điều chỉnh biên lề softmax y + tau_la * log(pi_i) để cân bằng phân phối nhãn.
  3. SCAFFOLD (chuẩn NVIDIA NVFlare): Control variates hiệu chỉnh gradient drift chỉ áp dụng lên
     các tham số Conv và Linear (non-BN parameters), thực thi độc lập ngoài momentum buffer.
"""

import time
import torch
import torch.nn as nn


def train_fedbn_scaffold_la(model, train_loader, optimizer, epochs, device,
                            c_global: list, c_local: list, lr: float,
                            log_priors: torch.Tensor = None, tau_la: float = 0.0):
    criterion = nn.CrossEntropyLoss()
    model.train()

    # Chỉ trích xuất các tham số trainable KHÔNG phải BatchNorm cho SCAFFOLD
    non_bn_params = [p for name, p in model.named_parameters() if "bn" not in name and p.requires_grad]

    # Khởi tạo biến tích lũy True Gradient (Option I - Karimireddy et al., ICML 2020)
    # Tách biệt hoàn toàn khỏi momentum buffer của optimizer
    grad_accum = [torch.zeros_like(p.data) for p in non_bn_params]

    c_g_dev = [cg.to(device) for cg in c_global]
    c_l_dev = [cl.to(device) for cl in c_local]

    use_la = (log_priors is not None) and (tau_la > 0.0)
    if use_la:
        log_priors_dev = log_priors.to(device)
    else:
        log_priors_dev = None

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

            # Logit Adjustment (chỉ kích hoạt nếu tau_la > 0.0)
            if use_la and log_priors_dev is not None:
                adjusted_outputs = outputs + tau_la * log_priors_dev
                loss = criterion(adjusted_outputs, labels)
            else:
                loss = criterion(outputs, labels)
            loss.backward()

            # Option I: Tích lũy trực tiếp true gradient của non-BN params ngay sau loss.backward()
            with torch.no_grad():
                for ga, p in zip(grad_accum, non_bn_params):
                    if p.grad is not None:
                        ga.add_(p.grad.data)

            optimizer.step()

            # Chuẩn NVIDIA NVFlare:
            # Hiệu chỉnh control variate trực tiếp lên các tham số non-BN SAU KHI optimizer.step()
            with torch.no_grad():
                for p, cg, cl in zip(non_bn_params, c_g_dev, c_l_dev):
                    p.data.add_(cg - cl, alpha=-lr)
            tau += 1

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    training_time = time.time() - start_time
    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0

    # 2. Cập nhật control variate cục bộ (Option I - True Gradient)
    # c_i_new = (1 / tau) * sum(grad_t)
    # delta_c = c_i_new - c_i_old
    c_local_new = []
    delta_c = []

    scale = 1.0 / float(tau) if tau > 0 else 1.0

    for cl, ga in zip(c_local, grad_accum):
        c_new = (ga * scale).cpu()
        dc = c_new - cl.to("cpu")
        c_local_new.append(c_new)
        delta_c.append(dc)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"   [FedBN-SCAFFOLD-LA Client] Steps: {tau} (LR: {current_lr:.6f}) | Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

    return {
        "c_local_new": c_local_new,
        "delta_c": delta_c,
        "tau": tau,
        "training_time": training_time,
        "num_samples": len(train_loader.dataset),
        "final_loss": epoch_loss,
        "final_accuracy": epoch_acc,
    }
