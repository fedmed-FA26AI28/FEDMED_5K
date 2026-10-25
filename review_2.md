# REVIEW 2: TỔNG QUAN HỆ THỐNG FEDERATED LEARNING & LỰA CHỌN NỀN TẢNG
*(Tài liệu chuẩn bị nội dung Slide Báo cáo & Bảo vệ Đề tài)*

> **Tài liệu tham chiếu cốt lõi:**  
> Peter Kairouz, H. Brendan McMahan et al., *"Advances and Open Problems in Federated Learning"*, Foundations and Trends in Machine Learning, 2021 (arXiv:1912.04977).

---

## SLIDE 1: Phân Loại Đa Chiều Các Hệ Thống Federated Learning (FL Taxonomy)

### 1.1. Architecture & Topology (Kiến trúc & Tô-pô mạng)
* **Centralized FL (Server–Client / Hub-and-Spoke)**:
  - **Cơ chế**: Có máy chủ trung tâm (*Parameter Server*) điều phối: gửi mô hình toàn cục, lựa chọn client, và tổng hợp trọng số (*Aggregation*).
  - **Ưu điểm**: Tốc độ hội tụ nhanh, dễ kiểm soát bảo mật và đồng bộ tham số.
  - **Hạn chế**: Điểm chết đơn độc (*Single Point of Failure*) và nghẽn băng thông máy chủ (*Bandwidth Bottleneck*).
  - *Áp dụng y tế*: Phù hợp nhất cho liên minh các bệnh viện có trung tâm điều phối quản trị.
* **Decentralized FL (Peer-to-Peer / Gossip-based)**:
  - **Cơ chế**: Không có máy chủ trung tâm. Các client trao đổi trực tiếp với các node láng giềng theo cấu trúc đồ thị mạng (*Graph Topology*).
  - **Ưu điểm**: Khử hoàn toàn điểm chết đơn độc, tăng tính tự trị giữa các bên.
  - **Hạn chế**: Tốc độ hội tụ chậm, chi phí truyền thông topo mạng cao, khó kiểm soát chất lượng mô hình.

---

### 1.2. Deployment Scale: Cross-Silo FL vs. Cross-Device FL *(Table 1 - Kairouz et al.)*

| Tiêu Chí So Sánh | Cross-Silo FL *(Đồ án FedMedAI)* | Cross-Device / Edge FL |
| :--- | :--- | :--- |
| **Đối tượng tham gia (Clients)** | **2 – 100 Tổ chức / Bệnh viện** | Hàng triệu thiết bị di động / IoT |
| **Tài nguyên tính toán** | **Mạnh**: Máy chủ y tế, GPU chuyên dụng | **Yếu**: Chip di động, pin/RAM giới hạn |
| **Tính sẵn sàng & Trạng thái** | **Gần 100% online liên tục** *(Stateful)* | <1% online/round, tỷ lệ rớt mạng cao *(Stateless)* |
| **Nghẽn cổ chai chính** | Chi phí tính toán / Xác thực pháp lý | Băng thông mạng truyền thông *(Communication)* |
| **Mục tiêu ứng dụng** | **Chẩn đoán y khoa chính xác cao** | Gợi ý từ bàn phím, đề xuất nội dung cá nhân |

> **Khẳng định trên Slide:** Đồ án **FedMedAI** thuộc nhóm **Cross-Silo Medical FL** (kết nối liên viện) và có thể mở rộng xuống **Edge Medical FL** (các máy trạm nhúng AI tại phòng khám cơ sở như NVIDIA Jetson).

---

### 1.3. Phân Vùng Dữ Liệu: Các Dạng Lệch Dữ Liệu Non-IID Trong Y Tế

* **1. Lệch tỷ lệ bệnh giữa các viện (Label Distribution Skew)**:
  - Tỷ lệ các nhóm bệnh rất khác nhau giữa các bệnh viện.
  - *Ví dụ y tế*: Bệnh viện Ung bướu tiếp nhận đa số bệnh nhân ung thư; Bệnh viện Đa khoa chủ yếu tiếp nhận bệnh nhân thông thường và rất ít ca ung thư.
* **2. Lệch đặc trưng ảnh do thiết bị chụp (Feature Distribution Skew / Scanner Bias)**:
  - Cùng một loại bệnh nhưng hình ảnh chụp ra có độ sáng, độ tương phản và độ nhiễu khác nhau do các bệnh viện dùng máy chụp của các hãng khác nhau (Siemens, GE, Philips).
* **3. Lệch tiêu chuẩn chẩn đoán & biểu hiện bệnh (Concept Drift)**:
  - *Cùng bệnh nhưng biểu hiện khác nhau*: Cùng tổn thương phổi nhưng hình ảnh X-quang ở người cao tuổi khác nhiều so với trẻ em.
  - *Cùng hình ảnh nhưng chẩn đoán khác nhau*: Cùng một lát cắt mô bệnh học nhưng ranh giới phân loại khối u giữa các trường phái y tế có độ vênh nhất định.
* **4. Lệch số lượng ca bệnh (Quantity Skew)**:
  - Quy mô dữ liệu chênh lệch rất lớn: Bệnh viện tuyến Trung ương có hàng trăm nghìn ca bệnh, trong khi phòng khám cơ sở chỉ có vài chục ca.
* **5. Phân Vùng Dữ Liệu Thực Tế Trong Đồ Án FedMedAI**:
  - **Bộ dữ liệu sử dụng**: Thực nghiệm trọng tâm trên **OrganMNIST-Axial** (11 nhóm cơ quan CT ổ bụng, 58 nghìn ảnh); mở rộng và đối chiếu với **BloodMNIST** (8 nhóm tế bào máu) và **DermaMNIST** (7 nhóm bệnh da).
  - **Dạng lệch dữ liệu trọng tâm**: Tập trung giải quyết **Lệch tỷ lệ bệnh giữa các viện (Label Distribution Skew)** — bài toán cốt lõi mà phương pháp đề xuất FedSLD (chia sẻ phân phối nhãn và hiệu chỉnh phân loại) giải quyết triệt để.
  - **Kịch bản thực nghiệm (12 bệnh viện / clients)**:
    1. *Kịch bản Thực tế (Practical Non-IID)*: Mô phỏng các bệnh viện đa khoa cùng tiếp nhận nhiều loại bệnh nhưng với tỷ lệ ca bệnh chênh lệch lớn giữa các viện.
    2. *Kịch bản Chuyên khoa Cực đoan (Pathological Non-IID)*: Mỗi bệnh viện chỉ tiếp nhận 1 đến 2 nhóm bệnh chuyên sâu, tạo thử thách tối đa để kiểm tra độ ổn định của thuật toán.

---

## SLIDE 2: Nhóm Hệ Thống Thuật Toán & Tổng Hợp (Algorithmic Systems)

### Nhánh 1 — Khử Client Drift & Lệch Nhãn (Thiên Về Xử Lý Phía Server / Global-Centric)
* **Bản chất & Kiến trúc**: Mục tiêu tối thượng là xây dựng **MỘT MÔ HÌNH TOÀN CỤC DUY NHẤT (Single Global Model)**. Do đó, **trọng tâm tối ưu nằm ở phía MÁY CHỦ (Server-side)**: Server tận dụng "góc nhìn toàn cảnh" (Global View) để điều phối, khử trôi dạt gradient (Client Drift), chuẩn hóa trọng số tổng hợp hoặc chia sẻ tri thức toàn cục.
* **Các giải thuật tiêu biểu**:
  - **FedAvg** *(McMahan et al., 2017)*: Baseline kinh điển, Server trung bình trọng số theo số lượng mẫu. Dễ mất ổn định khi Non-IID cao.
  - **FedProx** *(Li et al., 2020)*: Thêm thành phần điều chuẩn Proximal Term $\frac{\mu}{2} \|w - w_t\|^2$ ghìm khoảng cách giữa trọng số client và mô hình Server toàn cục; chịu lỗi tốt khi có client tính toán chậm (*stragglers*).
  - **SCAFFOLD** *(Karimireddy et al., 2020)*: Server duy trì và cập nhật biến đối chứng toàn cục (*Global Control Variates*) để ước lượng hướng trôi dạt trung bình và bù trừ gradient cho các client.
  - **FedNova** *(Wang et al., 2020)*: Server chuẩn hóa và tái cân bằng trọng số tổng hợp theo số bước tính toán thực tế của từng client, tránh thiên lệch về phía client chạy nhiều bước.
  - **Server Momentum & Adaptive FL** *(FedAvgM, FedAdam)*: Server coi vector tổng hợp như một gradient giả và tối ưu hóa bước cập nhật toàn cục bằng momentum / adaptive learning rate.
  - **FedSLD** *(Luo & Wu, ISBI 2022 - Đề tài tham khảo)*: Server tổng hợp tỷ lệ các nhóm bệnh thành **Phân Phối Nhãn Toàn Cục (Shared Label Distribution)** rồi phát xuống để từng client căn chỉnh hàm mất mát (*Logit Adjustment*), giải quyết triệt để vấn đề lệch nhãn trong ảnh y tế.

---

### Nhánh 2 — Xử Lý Lệch Đặc Trưng & Cá Nhân Hóa (Thiên Về Xử Lý Phía Client / Client-Centric)
* **Bản chất & Kiến trúc**: Thừa nhận rằng một mô hình toàn cục duy nhất không thể tối ưu cho mọi bệnh viện do khác biệt thiết bị chụp (**Scanner Bias**) và cơ địa bệnh nhân theo vùng miền. Do đó, **trọng tâm tối ưu nằm ở phía BỆNH VIỆN (Client-side)**: Từng client chủ động giữ lại hoặc tinh chỉnh các thành phần mô hình cho riêng mình thay vì ép buộc hội tụ tuyệt đối.
* **Các hướng tiếp cận tiêu biểu**:
  - **Local Fine-Tuning**: Sau khi huấn luyện mô hình toàn cục qua FL, mỗi client tự chạy thêm vài epoch tinh chỉnh trên dữ liệu nội bộ của riêng mình.
  - **Model Splitting (FedRep, FedPer)**: Client giữ riêng đầu phân loại cục bộ (*Personalized Head*) phù hợp cơ cấu bệnh viện, chỉ gửi phần trích xuất đặc trưng chung (*Shared Backbone*) lên Server.
  - **Domain Normalization (FedBN - Li et al., 2021)**: Client giữ riêng các lớp Batch Normalization cục bộ để triệt tiêu nhiễu ánh sáng/độ tương phản máy chụp, chỉ chia sẻ các tầng Convolution lên Server.
  - **Meta-Learning (Per-FedAvg, pFedHN)**: Áp dụng tư tưởng MAML: Huấn luyện mô hình khởi tạo sao cho mỗi client chỉ cần 1 vài bước gradient cục bộ là thích ứng tối ưu nhất với dữ liệu riêng.

---

## SLIDE 3: Bảng So Sánh Đa Chiều Nền Tảng Phần Mềm & Lý Do Lựa Chọn Flower (flwr)

| Nền Tảng FL | Khả Năng Tùy Biến Thuật Toán (Custom Strategy) | Mức Độ Dễ Học & Tiếp Cận (Learning Curve) | Mô Phỏng & Triển Khai Thực Tế (Simulation to Edge) |
| :--- | :--- | :--- | :--- |
| **NVIDIA FLARE**<br>*(NVIDIA)* | • **Phức tạp & Rườm rà**: Kiến trúc Event-driven Controller/Executor đòi hỏi nhiều mã nguồn khung (boilerplate), rất khó can thiệp sâu để thử nghiệm giải thuật mới. | • **Rất khó tiếp cận**: Đường cong học tập dốc, cấu hình hệ thống nặng nề, phụ thuộc chặt chẽ vào phần cứng GPU và hệ sinh thái của NVIDIA. | • Tối ưu mạnh cho cụm GPU máy chủ lớn; triển khai xuống thiết bị nhúng nhỏ (Jetson) đòi hỏi cấu hình container cồng kềnh. |
| **OpenFL**<br>*(Intel / Linux Fdn)* | • **Cứng nhắc & Hạn chế**: Quy trình tổng hợp bị đóng khung theo chuẩn bảo mật, khó chỉnh sửa logic tổng hợp lõi hoặc áp dụng các hàm loss hiệu chỉnh mới. | • **Khó tiếp cận**: Đòi hỏi thiết lập hạ tầng chứng chỉ bảo mật (PKI/TLS) phức tạp ngay từ đầu; tài liệu ít và cộng đồng hỗ trợ hạn chế. | • Tập trung tốt cho kết nối liên viện Cross-Silo; khả năng mô phỏng linh hoạt nhiều client trên 1 máy tính đơn lẻ còn yếu. |
| **Fed-BioMed**<br>*(INRIA Pháp)* | • **Rất khó mở rộng**: Chủ yếu đóng gói sẵn một số thuật toán cơ bản, không được thiết kế tối ưu cho việc tùy biến và nghiên cứu thuật toán mới. | • **Khó tiếp cận**: Bắt buộc cài đặt Docker và cấu hình node bệnh viện ảo, nặng về kỹ năng quản trị hạ tầng IT hơn là nghiên cứu ML. | • Chuẩn mực tốt cho thử nghiệm lâm sàng thực tế tại bệnh viện, nhưng tốc độ mô phỏng thử nghiệm học thuật rất chậm. |
| **Flower (flwr)**<br>★ *(Lựa chọn của đồ án)* | • **Tối ưu & Linh hoạt nhất**: Kế thừa trực tiếp `Strategy`, override hàm `aggregate_fit` là hiện thực hóa FedProx, FedNova, FedSLD chỉ với vài chục dòng code. | • **Thân thiện nhất cho người mới**: Cực kỳ lý tưởng cho người mới nghiên cứu, học tập FL; API Pythonic trực quan, tài liệu phong phú, chạy ngay trên PC/Laptop. | • **Chuyển tiếp liền mạch (100% code reuse)**: Chạy mô phỏng 10–50 viện trên 1 GPU (Virtual Client Engine) rồi mang nguyên code sang cụm Jetson / Bệnh viện thực tế. |
