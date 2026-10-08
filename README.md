Dưới đây là bản thảo cho tệp README.md. Bản hướng dẫn này được thiết kế với văn phong kỹ thuật chuẩn mực, rõ ràng, giúp bất kỳ ai đọc vào cũng có thể tự tay thiết lập và vận hành hệ thống phần cứng một cách trơn tru.

# Triển Khai Federated Learning  Trên Cụm NVIDIA Jetson

Tài liệu này hướng dẫn chi tiết quy trình triển khai thuật toán Học liên kết (Federated Learning) sử dụng framework Flower, huấn luyện mạng CNN trên bộ dữ liệu y tế BloodMNIST với các thiết bị biên là NVIDIA Jetson.

---

## Research simulation (controlled 5K regime)

For the validation-only experiment matrix, native 64×64 data, model comparisons,
locked final-test procedure, and Jetson measurements, see [RESEARCH_STUDY.md](RESEARCH_STUDY.md).

The simulator now selects exactly 5,000 stratified examples from the official
training split before applying the non-IID partition. It never partitions
validation or test examples as training data.

```powershell
python -m experiments.run_simulation --strategy coverage --num_clients 10 --alpha 0.3 --rounds 30 --local_epochs 1 --train_samples 5000 --size 64 --model tiny_cnn
```

For a fair ablation, keep `--seed`, client count, alpha, rounds, and data budget
identical while running `fedavg`, `coverage --head_mu 0`, `coverage
--logit_tau 0`, and the full `coverage` configuration.

Half of validation is used for client monitoring and exact final-model
validation; the other half is held out for temperature/conformal calibration.
Development runs do not load test. Only `--final_test --locked_config RUN_SPEC`
loads test once after training and reports final metrics.

## 1. Yêu Cầu Hệ Thống

| Thành phần | Yêu cầu chi tiết |
| --- | --- |
| **Máy chủ (Server)** | 01 Laptop/PC (Windows/Linux/macOS) |
| **Máy khách (Clients)** | 03 bo mạch NVIDIA Jetson Nano/Orin |
| **Hạ tầng mạng** | 01 Router Wi-Fi/LAN nội bộ |
| **Thiết bị phụ trợ** | 01 USB Flash Drive |
| **Phần mềm lõi** | Python 3.8+, thư viện `flwr`, `medmnist`, `scikit-learn` |
| **Môi trường Jetson** | PyTorch phiên bản hỗ trợ CUDA (Tối ưu GPU Maxwell) |

## 2. Tiền Xử Lý Dữ Liệu Ngoại Tuyến (Laptop)

### Mô phỏng nhiều client trên một máy

```powershell
# Chạy thử nhanh
python -m experiments.run_simulation --num_clients 5 --rounds 2 --local_epochs 1 --strategy fedavg

# Chạy 10 clients với dữ liệu non-IID
python -m experiments.run_simulation --num_clients 10 --rounds 50 --local_epochs 5 --alpha 0.3 --strategy fedavg
```

Mỗi client dùng validation riêng; test chỉ được đánh giá trong final run có cấu hình khóa.

* Chạy tệp mã nguồn `prepare_data.py` trên Laptop để tự động tải và chia bộ dữ liệu BloodMNIST theo phân phối Dirichlet.
* Kiểm tra thư mục `data` vừa được tạo ra để đảm bảo có đủ các tệp nhị phân dữ liệu cho từng máy khách.
* Chép toàn bộ thư mục `data` cùng hai tệp mã nguồn là `common.py` và `client.py` vào USB.

## 3. Triển Khai Vật Lý (Thiết Bị Jetson)

* Cắm USB vào từng bo mạch Jetson và tạo một thư mục dự án cục bộ mới.
* Chép hai tệp `common.py` và `client.py` vào thư mục dự án vừa tạo trên Jetson.
* Chép đúng thư mục dữ liệu cá nhân hóa (ví dụ: chỉ chép `data/client_0`) từ USB sang thiết bị Jetson số 0 để đảm bảo tính phân tán dữ liệu khắt khe.
* Mở terminal trên thiết bị Jetson và chạy lệnh cài đặt các thư viện cần thiết nếu môi trường chưa có sẵn.

Làm việc với phần cứng nhúng (như Jetson hay Raspberry Pi) lần đầu tiên đúng là sẽ hơi bỡ ngỡ vì chúng chạy hệ điều hành Ubuntu thuần túy và thường giao tiếp qua dòng lệnh. Tuy nhiên, bản chất nó chỉ là một chiếc máy tính thu nhỏ.

Để bạn hoàn toàn tự tin khi thao tác thật, mình đã viết lại Phần 4 "cầm tay chỉ việc" chi tiết nhất có thể. Bạn có thể thay thế phần này vào tệp README:

---

## 4. Vận Hành Hệ Thống Xuyên Mạng (Live Execution) chi tiết

**4.1. Cố định địa chỉ mạng cho Server (Laptop)**

* Đảm bảo Laptop và tất cả các máy Jetson đều đang kết nối vào chung một cục phát Wi-Fi.
* Trên Laptop (Windows), mở ứng dụng **Command Prompt** (cmd) và gõ lệnh `ipconfig`.
* Tìm dòng **IPv4 Address** (ví dụ: `192.168.1.15`). Hãy ghi nhớ dãy số này vì đây là "tọa độ" để các máy Jetson tìm về Laptop.
* Tắt tạm thời Windows Defender Firewall (hoặc phần mềm diệt virus) trên Laptop để các thiết bị bên ngoài có thể gửi dữ liệu vào cổng 8080.

**4.2. Giao tiếp và sửa mã nguồn trên Jetson**
Vì Jetson là một máy tính độc lập, bạn có hai cách để thao tác với nó:

* **Cách dễ nhất:** Cắm một màn hình vào cổng HDMI của Jetson, cắm chuột và bàn phím qua cổng USB. Thao tác trên giao diện màn hình y hệt như dùng một chiếc máy tính bình thường.
* **Cách chuyên nghiệp (SSH):** Mở terminal trên Laptop và gõ lệnh `ssh tên_đăng_nhập@IP_của_Jetson` để điều khiển Jetson từ xa (không cần cắm thêm màn hình ngoài).
* Bất kể dùng cách nào, hãy mở terminal trên Jetson, đi đến thư mục chứa tệp `client.py` và dùng lệnh chỉnh sửa văn bản (ví dụ: gõ `nano client.py`).
* Tìm dòng khai báo địa chỉ máy chủ và thay thế bằng IP của Laptop mà bạn vừa lấy ở Bước 4.1 (ví dụ: `server_address="192.168.1.15:8080"`). Bấm Ctrl+O, Enter để lưu và Ctrl+X để thoát. Lặp lại bước này cho tất cả các máy Jetson.

**4.3. Thứ tự khởi chạy hệ thống (Cực kỳ quan trọng)**

* **Khởi động Trạm Chỉ Huy:** Trên Laptop, mở terminal trong thư mục dự án và gõ lệnh: `python server.py`. Màn hình sẽ hiện thông báo máy chủ đang lắng nghe trên cổng 8080 và chặn luồng (đứng yên) để chờ.
* **Đánh thức các Điểm Biên:** Lần lượt di chuyển sang các máy Jetson, mở terminal và gõ lệnh: `python client.py`.
* Khi máy Jetson cuối cùng gõ xong lệnh, Server sẽ nhận diện đủ số lượng thiết bị và ngay lập tức gửi tập lệnh bắt đầu Vòng 1.

**4.4. Theo dõi diễn biến huấn luyện**

* Trên màn hình của các máy Jetson, bạn sẽ thấy thông báo nhận trọng số và bắt đầu tải các batch dữ liệu (thể hiện qua thanh tiến trình Train/Loss). Lõi tản nhiệt của Jetson sẽ bắt đầu quay mạnh vì GPU đang chạy hết công suất.
* Trên màn hình Laptop, sau mỗi vòng giao tiếp, hệ thống sẽ in ra thông số độ chính xác toàn cục (Global Metrics) và dòng chữ thông báo đã lưu tệp `.pth`.

---

