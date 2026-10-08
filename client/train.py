"""Chứa vòng lặp huấn luyện cục bộ (local training loop). Thực hiện forward/backward pass để cập nhật trọng số mô hình."""


import torch
import torch.nn as nn
import time
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from algorithms.coverage import coverage_head_penalty, logit_adjustment

def train(
    model,
    train_loader,
    optimizer,
    epochs,
    device,
    val_loader=None,
    min_lr: float = 1e-6,
    class_counts=None,
    logit_tau: float = 0.0,
    prior_smoothing: float = 1.0,
    head_mu: float = 0.0,
    coverage_kappa: float = 32.0,
    global_head=None,
    proximal_mu: float = 0.0,
    global_params=None,
):

    criterion = nn.CrossEntropyLoss()
    adjustment = None
    if class_counts is not None and logit_tau > 0:
        adjustment = logit_adjustment(
            class_counts, tau=logit_tau, smoothing=prior_smoothing
        ).to(device)
    model.train()

    start_time = time.time()

    # 1. Khởi tạo Learning Rate Scheduler
    # - factor=0.5: giảm LR xuống còn 1/2 khi Val Loss không cải thiện
    # - patience=3: chờ 3 epoch trước khi giảm LR
    # - min_lr: ngưỡng sàn, LR sẽ không bao giờ giảm xuống dưới giá trị này
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=min_lr)
    
    # 2. Khai báo biến cho Early Stopping
    best_loss = float('inf')
    patience_counter = 0
    patience = 8
    best_weights = None
    best_epoch = 0        # Lưu lại số thứ tự epoch tốt nhất
    best_val_loss = None  # Val Loss tại epoch tốt nhất
    best_val_acc  = None  # Val Accuracy tại epoch tốt nhất

    history ={
        "loss": [],
        "accuracy": []
    }

    if val_loader is not None:
        history["val_loss"] = []
        history["val_accuracy"] = []
    
    total = 0
    for epoch in range(epochs):
        # Bắt buộc phải có model.train() ở đầu mỗi Epoch
        model.train() 
        
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.squeeze(-1).long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            adjusted_outputs = (
                outputs + adjustment.unsqueeze(0)
                if adjustment is not None
                else outputs
            )
            loss = criterion(adjusted_outputs, labels)
            if head_mu > 0:
                loss = loss + coverage_head_penalty(
                    model,
                    global_head,
                    class_counts,
                    head_mu=head_mu,
                    kappa=coverage_kappa,
                )
            if proximal_mu > 0:
                loss = loss + 0.5 * proximal_mu * sum(
                    (current - reference).pow(2).sum()
                    for current, reference in zip(model.parameters(), global_params)
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() *images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_accuracy = correct / total

        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_accuracy)
        
        #Kẹp Learning Rate hiện tại vào chuỗi thông báo
        current_lr = optimizer.param_groups[0]['lr']
        print_str = f"Epoch [{epoch + 1}/{epochs}] (LR: {current_lr:.6f}) | Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}"

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for val_images, val_labels in val_loader:
                    val_images = val_images.to(device)
                    val_labels = val_labels.squeeze(-1).long().to(device)

                    val_outputs = model(val_images)
                    v_loss = criterion(val_outputs, val_labels)
                    val_loss += v_loss.item() * val_images.size(0)

                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_total += val_labels.size(0)
                    val_correct += (val_predicted == val_labels).sum().item()
            val_epoch_loss = val_loss / val_total
            val_epoch_accuracy = val_correct / val_total
            history['val_loss'].append(val_epoch_loss)  
            history['val_accuracy'].append(val_epoch_accuracy)

            print_str += f" | Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.4f}"
            print(print_str)
            
            
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_epoch_loss)
            new_lr = optimizer.param_groups[0]['lr']
            
            if new_lr < old_lr:
                print(f"   Val Loss không giảm, tự động giảm Learning Rate xuống {new_lr}")
                
                
            # --- ÁP DỤNG EARLY STOPPING DỰA TRÊN VAL LOSS ---
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                patience_counter = 0
                best_weights = copy.deepcopy(model.state_dict())
                # Lưu lại thông số của epoch tốt nhất cùng với best_weights
                best_epoch    = epoch + 1
                best_val_loss = val_epoch_loss
                best_val_acc  = val_epoch_accuracy
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"--- Early Stopping kích hoạt ở Epoch {epoch + 1}! Val Loss không giảm trong {patience} vòng liên tiếp. ---")
                    break
        else:
            # Nếu không có tập Val, chỉ in kết quả của Train
            print(print_str)
        
    training_time = time.time() - start_time
    # [LƯU Ý: ĐOẠN CODE DƯỚI ĐÂY DÙNG KHI CHẠY CENTRALIZED]
    # Trong FL giả lập, chúng ta trả về model của epoch cuối cùng cho Server
    # if val_loader is not None and best_weights is not None:
    #     print("Đang nạp lại trọng số tốt nhất (Best Weights) để chuẩn bị đi Test...")
    #     model.load_state_dict(best_weights)

    # [FIX 2] Tách riêng val_loss/val_accuracy ra ngoài để tránh KeyError khi không có val_loader
    metrics = {
        "training_time": training_time,
        "final_loss": history['loss'][-1] if history['loss'] else 0.0,
        "final_accuracy": history['accuracy'][-1] if history['accuracy'] else 0.0,
        "num_samples": total,
        "history": history
    }
    # Chỉ thêm val metrics vào nếu thực sự có val_loader
    if val_loader is not None and 'val_loss' in history and history['val_loss']:
        metrics["final_val_loss"]     = history['val_loss'][-1]
        metrics["final_val_accuracy"] = history['val_accuracy'][-1]
        # Thông số tại epoch tốt nhất (chính là best_weights đang được load)
        metrics["best_epoch"]    = best_epoch
        metrics["best_val_loss"] = best_val_loss
        metrics["best_val_acc"]  = best_val_acc
    return metrics
