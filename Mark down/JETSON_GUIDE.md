# 🚀 Hướng Dẫn Triển Khai Thực Tế — PC Server + Jetson Client

Tài liệu này hướng dẫn chi tiết quy trình kết nối **PC (Windows) làm Server** và **NVIDIA Jetson làm Client** trong hệ thống Federated Learning sử dụng framework **Flower + BloodMNIST + SimpleCNN**.

---

## 1. Kiến Trúc Triển Khai

```
┌─────────────────────────────────────────────────────────┐
│                      Wi-Fi / LAN                        │
│                                                         │
│   ┌─────────────┐     gRPC :8080     ┌───────────────┐  │
│   │  PC (Server)│ ◄─────────────────►│ Jetson (Client│  │
│   │  Windows    │                    │  Orin / Nano) │  │
│   │  run_server │                    │  run_jetson_  │  │
│   │  .py        │                    │  client.py    │  │
│   └─────────────┘                    └───────────────┘  │
└─────────────────────────────────────────────────────────┘
```

| Vai trò | Thiết bị | Script |
|---|---|---|
| **Server** | PC / Laptop (Windows) | `experiments/run_server.py` |
| **Client** | NVIDIA Jetson (Orin/Nano) | `experiments/run_jetson_client.py` |
| **Monitor** | Chạy trên Jetson (background) | `client/system_monitor.py` |

---

## 2. Yêu Cầu Hệ Thống

| Thành phần | Chi tiết |
|---|---|
| **PC Server** | Windows 10/11, Python 3.9+, GPU NVIDIA (tùy chọn) |
| **Jetson Client** | JetPack 5.x+, Python 3.8+, CUDA tích hợp |
| **Mạng** | Cùng router Wi-Fi / LAN nội bộ |
| **Thư viện chung** | `flwr`, `medmnist`, `torch`, `pyyaml`, `scikit-learn` |
| **Thư viện Jetson thêm** | `psutil` (để monitor tài nguyên) |

---

## 3. Chuẩn Bị Trên PC (Tối Trước)

### 3.1 Tạo partition dữ liệu

Nếu chưa chạy Phase 4 (phân vùng dữ liệu), hãy chạy:

```bash
# Trên PC — tạo file partition cho 1 client
python -m experiments.run_partition --num_clients 1 --alpha 1.0
```

File kết quả sẽ nằm tại:
```
data/partitions/partition_seed42_alpha1.0_clients1.json
```

### 3.2 Tìm địa chỉ IP của PC

Mở **PowerShell** và gõ:

```powershell
ipconfig
```

Tìm dòng **IPv4 Address** trong mục **Wi-Fi** (ví dụ: `192.168.1.10`). Ghi nhớ địa chỉ này.

### 3.3 Mở port 8080 trên Windows Firewall

```powershell
# Chạy PowerShell với quyền Administrator
netsh advfirewall firewall add rule name="Flower FL" dir=in action=allow protocol=TCP localport=8080
```

> **Lưu ý:** Nếu không muốn mở Firewall lâu dài, có thể tạm thời tắt Windows Defender Firewall trong thời gian thử nghiệm.

---

## 4. Chuẩn Bị Trên Jetson

### 4.1 Clone project lên Jetson

**Cách 1 — SSH từ PC:**
```bash
# Mở PowerShell trên PC
ssh jetson@<IP_CUA_JETSON>
# Ví dụ: ssh jetson@192.168.1.50
```

**Cách 2 — Cắm màn hình + bàn phím trực tiếp vào Jetson** và mở Terminal.

Sau khi vào được terminal Jetson:
```bash
# Clone repo (hoặc copy thủ công qua USB/SCP)
git clone <URL_REPO> ~/FedMedAI
cd ~/FedMedAI
```

### 4.2 Cài đặt thư viện trên Jetson

```bash
pip install flwr medmnist pyyaml scikit-learn psutil
# PyTorch thường đã được cài sẵn theo JetPack
# Nếu chưa có, xem: https://forums.developer.nvidia.com/t/pytorch-for-jetson
```

### 4.3 Gửi dữ liệu partition từ PC sang Jetson

Trên **PC**, cập nhật IP Jetson trong `scripts/distribute_data.py`:
```python
JETSON_CLIENTS = [
    {
        "client_id": 0,
        "ip": "192.168.1.50",       # <-- Đổi thành IP thực của Jetson
        "username": "jetson",        # <-- Đổi thành username của Jetson
        "remote_dir": "~/FedMedAI/data/"
    },
    ...
]
```

Sau đó chạy lệnh gửi dữ liệu:
```bash
# Trên PC — tự động SCP file sang Jetson
python -m scripts.distribute_data --client_id 0
```

> Script sẽ tự ping kiểm tra kết nối, tạo thư mục trên Jetson và gửi file partition qua SCP.

---

## 5. Vận Hành Hệ Thống (Ngày Thử Nghiệm)

> ⚠️ **Thứ tự quan trọng:** Luôn khởi động **Server TRƯỚC**, sau đó mới khởi động **Client**.

### 5.1 Khởi động Server (trên PC)

```bash
# Trên PC — Terminal 1
python -m experiments.run_server \
    --strategy fedavg \
    --num_clients 1 \
    --rounds 20 \
    --port 8080
```

Màn hình sẽ hiển thị:
```
============================================================
  FL SERVER DANG KHOI DONG
  Strategy   : FEDAVG
  Clients    : 1
  Rounds     : 20
  Listening  : 0.0.0.0:8080
  >>> Dang cho Jetson Client ket noi... <<<
============================================================
```

Server sẽ **đứng yên chờ** cho đến khi Jetson kết nối vào.

### 5.2 Khởi động Monitor (trên Jetson) — Tùy chọn

```bash
# Trên Jetson — Terminal 1 (chạy nền)
python -m client.system_monitor \
    --output results/jetson_monitor.json \
    --interval 5
```

### 5.3 Khởi động Client (trên Jetson)

```bash
# Trên Jetson — Terminal 2
python experiments/run_jetson_client.py \
    --server_ip 192.168.1.10 \
    --client_id 0 \
    --alpha 1.0
```

> Thay `192.168.1.10` bằng **IP thực của PC** lấy ở Bước 3.2.

### 5.4 Theo Dõi Kết Quả

**Trên màn hình PC (Server)** sẽ in ra sau mỗi round:
```
  [Round   1] Acc=0.5234 Pre=0.4912 Rec=0.4801 F1=0.4720 LR=0.001000
  [Round   2] Acc=0.6891 Pre=0.6543 Rec=0.6312 F1=0.6290 LR=0.001000
  ...
```

**Trên màn hình Jetson (Client)** sẽ in ra quá trình train:
```
  Epoch [1/5] (LR: 0.001000) | Train Loss: 1.2341, Train Acc: 0.6123 | Val Loss: 0.9812, Val Acc: 0.6891
  Epoch [2/5] (LR: 0.001000) | Train Loss: 0.9123, Train Acc: 0.7234 | ...
```

---

## 6. Các Tham Số Dòng Lệnh

### `run_server.py`

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--strategy` | `fedavg` | Thuật toán FL: `fedavg`, `fedprox`, `fedavgm`, `fedmedian`, `fedtrimmedavg` |
| `--num_clients` | `1` | Số lượng Jetson sẽ kết nối |
| `--rounds` | `50` | Số round FL |
| `--host` | `0.0.0.0` | IP lắng nghe (mặc định là tất cả interface) |
| `--port` | `8080` | Cổng gRPC |

### `run_jetson_client.py`

| Tham số | Bắt buộc | Mô tả |
|---|---|---|
| `--server_ip` | ✅ | IP của PC Server (ví dụ: `192.168.1.10`) |
| `--port` | `8080` | Cổng gRPC (phải khớp với Server) |
| `--client_id` | `0` | ID của client (0–9, dùng để lấy partition đúng) |
| `--alpha` | `1.0` | Mức Non-IID của partition: `1.0`, `0.3`, `0.1` |
| `--device_type` | tự động | Ghi đè loại thiết bị: `jetson_orin` hoặc `jetson_nano` |

---

## 7. Cấu Hình Phần Cứng Tự Động

Hệ thống tự động đọc `configs/jetson.yaml` để chọn cấu hình phù hợp cho từng thiết bị:

| Thiết bị | `batch_size` | `num_workers` | `pin_memory` |
|---|---|---|---|
| **Jetson Orin** | 32 | 2 | `false` |
| **Jetson Nano** | 16 | 0 | `false` |
| **PC Simulation** | 32 | 2 | `true` |

> **Tại sao Jetson dùng `pin_memory=false`?** Jetson sử dụng kiến trúc **Unified Memory** (CPU và GPU chia sẻ cùng bộ nhớ RAM vật lý), nên không cần pin memory như PC thông thường.

---

## 8. Xử Lý Sự Cố

| Lỗi | Nguyên nhân | Giải pháp |
|---|---|---|
| `Connection refused` trên Jetson | PC Firewall chặn cổng 8080 | Xem Bước 3.3 |
| `FileNotFoundError: partition_seed42...` | Chưa gửi file partition sang Jetson | Chạy `distribute_data.py` (Bước 4.3) |
| `CUDA out of memory` trên Nano | Batch size quá lớn | Sửa `batch_size: 8` trong `jetson.yaml` mục `jetson_nano` |
| Jetson không ping được PC | Khác subnet | Đảm bảo cả hai cùng kết nối vào **1 router** |
| `timeout` khi SCP | SSH chưa được cấu hình | Chạy `ssh-keygen` và `ssh-copy-id jetson@IP` để bỏ mật khẩu |

---

## 9. Kết Quả Đầu Ra

Sau khi hoàn thành, kết quả được lưu tại:

```
results/
├── simulate_federated/           ← Kết quả chạy giả lập trên PC (Phase 5)
│   ├── clients3_alpha1.0_fedavg/
│   │   └── results.json
│   └── comparison_table.csv
│
└── real_federated/               ← Kết quả chạy THỰC trên Jetson (Phase 6+)
    ├── real_clients1_alpha1.0_fedavg/
    │   └── results.json
    └── real_clients1_alpha0.3_fedprox/
        └── results.json
```

Kết quả monitor Jetson (nếu bật):
```
results/
└── jetson_monitor.json       ← CPU/GPU/RAM/Nhiệt độ theo thời gian thực
```
