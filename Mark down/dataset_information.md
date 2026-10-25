# THÔNG TIN BẢN QUYỀN, NGUỒN GỐC & TÍNH XÁC THỰC LÂM SÀNG CỦA CÁC BỘ DỮ LIỆU MEDMNIST
*(OrganMNIST, PathMNIST, BloodMNIST, DermaMNIST)*

---

## 1. Thông Tin Cơ Bản Về Bộ Chuẩn Benchmark MedMNIST

- **Tên đầy đủ**: MedMNIST v2 – A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification.
- **Công bố khoa học**: Xuất bản chính thức trên tạp chí hàng đầu **Nature Scientific Data** (Volume 10, Article 41, 2023) bởi nhóm nghiên cứu quốc tế do Jiancheng Yang, Rui Shi, Donglai Wei, Hanspeter Pfister (Đại học Harvard, Đại học Giao thông Thượng Hải, EPFL...) chủ trì.
- **Mục tiêu thiết kế**: 
  - Đóng vai trò như một **"MNIST trong y tế"**: Chuẩn hóa các dạng ảnh y sinh phức tạp (CT, X-quang, Tiêu bản mô học, Ảnh soi da, Kính hiển vi máu...) về cùng một kích thước chuẩn **28×28 pixels** (hoặc 28×28×28 cho 3D) với định dạng đơn kênh xám hoặc 3 kênh RGB.
  - Cho phép các nhà nghiên cứu thuật toán Machine Learning / Federated Learning thử nghiệm nhanh (rapid prototyping), benchmark công bằng, tái hiện kết quả (reproducible) mà không cần cấu hình siêu máy tính hàng trăm GPU.
- **Quy mô toàn bộ MedMNIST v2**: Gồm **18 bộ dữ liệu** (12 bộ 2D và 6 bộ 3D) với tổng cộng hơn 708,000 ảnh 2D và gần 10,000 khối thể tích 3D.
- **Quy tắc phân chia tập dữ liệu (Standard Split)**: Mọi tập dữ liệu đều được chia cố định thành `Train` / `Validation` / `Test` ở **cấp độ bệnh nhân (Patient-Level Split)**. Điều này loại bỏ hoàn toàn hiện tượng rò rỉ dữ liệu (Data Leakage) giữa tập huấn luyện và kiểm thử.
- **Bản quyền & Giấy phép**: Được phân phối theo giấy phép mở **Creative Commons Attribution 4.0 International (CC BY 4.0)** hoặc theo giấy phép mở của từng bộ dữ liệu nguồn gốc.

---

## 2. Nguồn Gốc & Mức Độ Legit (Thẩm Định Chuyên Gia Y Tế) Của Từng Tập

### 2.1. BloodMNIST (Kính hiển vi tế bào máu ngoại vi)

* **Phương thức ảnh (Modality)**: Ảnh kính hiển vi quang học tế bào máu ngoại vi (Microscopic peripheral blood smear images).
* **Công trình khoa học gốc**: Acevedo et al. (2020), *"A dataset of microscopic peripheral blood cell images for development of automatic recognition systems"*, công bố trên tạp chí **Data in Brief** (Elsevier), Volume 30, 105474. DOI: `10.1016/j.dib.2020.105474`.
* **Cơ sở y tế & Đơn vị thu thập**: 
  - **Core Laboratory, Bệnh viện Đại học Barcelona (Hospital Clínic de Barcelona, Tây Ban Nha)** – một trong những bệnh viện tuyến cuối hàng đầu châu Âu.
  - Phối hợp cùng Đại học Bách khoa Catalonia (Universitat Politècnica de Catalunya – UPC BarcelonaTech).
* **Thiết bị thu nhận**: Hệ thống chụp và phân tích tế bào tự động cao cấp **CellaVision DM96** – thiết bị chuẩn vàng đạt chứng chỉ xét nghiệm lâm sàng quốc tế (ISO).
* **Mức độ Legit & Chuyên gia thẩm định nhãn (Ground Truth Validation)**:
  - Dữ liệu thu từ mẫu máu ngoại vi thường quy của 100 bệnh nhân không có bất thường huyết học hay nhiễm trùng.
  - **100% hình ảnh tế bào đơn lẻ được kiểm tra, phân loại và xác thực trực tiếp bởi các Bác sĩ chuyên khoa Huyết học lâm sàng (Clinical Hematologists) và Bác sĩ Giải phẫu bệnh (Clinical Pathologists)** giàu kinh nghiệm tại Core Laboratory của Bệnh viện Hospital Clínic de Barcelona.
* **Thông số kỹ thuật**:
  - Kích thước: `28×28×3` (RGB).
  - Số lượng mẫu: **17,092 ảnh** (Train: 11,959 | Val: 1,712 | Test: 3,421).
  - Gồm **8 lớp tế bào máu**:
    1. Basophils (Bạch cầu ưa kiềm)
    2. Eosinophils (Bạch cầu ưa acid)
    3. Erythroblasts (Tiền hồng cầu có nhân)
    4. Immature granulocytes (Bạch cầu hạt chưa trưởng thành: promyelocyte, myelocyte, metamyelocyte)
    5. Lymphocytes (Bạch cầu lympho)
    6. Monocytes (Bạch cầu đơn nhân)
    7. Neutrophils (Bạch cầu trung tính)
    8. Platelets (Tiểu cầu / thrombocytes)

---

### 2.2. PathMNIST (Mô bệnh học ung thư đại trực tràng)

* **Phương thức ảnh (Modality)**: Hình ảnh giải phẫu bệnh mô học nhuộm Hematoxylin & Eosin (H&E-stained Histopathology).
* **Công trình khoa học gốc**:
  - Jakob Nikolas Kather et al., *"Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study"*, công bố trên tạp chí y khoa danh giá **PLOS Medicine** (2019), 16(1): e1002730. DOI: `10.1371/journal.pmed.1002730`.
  - Nghiên cứu liên quan trên **Nature Medicine** (2019): *"Deep learning can predict microsatellite instability directly from histology in gastrointestinal cancer"*.
* **Cơ sở y tế & Đơn vị thu thập**: 
  - **Trung tâm Quốc gia về Bệnh lý Khối u (National Center for Tumor Diseases - NCT), Bệnh viện Đại học Heidelberg (Heidelberg University Hospital, CHLB Đức)**.
  - Phối hợp cùng Trung tâm Nghiên cứu Ung thư Đức (German Cancer Research Center - DKFZ) và Trung tâm Y tế Đại học Mannheim.
* **Mức độ Legit & Chuyên gia thẩm định nhãn (Ground Truth Validation)**:
  - Mẫu bệnh phẩm lấy từ các khối mô đúc sáp paraffin cố định formalin (FFPE) thu thập từ các ca phẫu thuật cắt đại trực tràng của bệnh nhân ung thư.
  - **Được các Bác sĩ chuyên khoa Giải phẫu bệnh (Certified Board Pathologists) tại Viện Giải phẫu bệnh Bệnh viện Đại học Heidelberg** trực tiếp soi kính hiển vi, khoanh vùng và xác thực phân loại từng cấu trúc vi mô.
  - **Giá trị bảo chứng ngoại suy độc lập (External Cohort Validation)**: 
    - Tập `Train` và `Val` trích xuất từ tập `NCT-CRC-HE-100K` (86 bệnh nhân tại Heidelberg).
    - Tập `Test` (7,180 ảnh) trích xuất hoàn toàn từ tập độc lập `CRC-VAL-HE-7K` thuộc nghiên cứu dịch tễ đa trung tâm **DACHS** (Darmkrebs: Chancen der Verhütung durch Screening – gồm 50 bệnh nhân hoàn toàn mới).
    - **Zero Patient Overlap**: Tuyệt đối không có bất kỳ bệnh nhân nào trùng lặp giữa train và test. Đây là chuẩn mực vàng cao nhất trong y khoa nhằm kiểm chứng tính khái quát hóa lâm sàng.
* **Thông số kỹ thuật**:
  - Kích thước: `28×28×3` (RGB).
  - Số lượng mẫu: **107,180 ảnh** (Train: 89,996 | Val: 10,004 | Test: 7,180).
  - Gồm **9 lớp mô học**:
    1. Adipose (Mô mỡ)
    2. Background (Khoảng trống / nền tiêu bản)
    3. Debris (Mảnh vụn hoại tử / tế bào chết)
    4. Lymphocytes (Tế bào miễn dịch lympho)
    5. Mucus (Chất nhầy)
    6. Smooth muscle (Cơ trơn)
    7. Normal colon mucosa (Niêm mạc ruột lành tính)
    8. Cancer-associated stroma (Mô liên kết khối u)
    9. Colorectal adenocarcinoma epithelium (Biểu mô ung thư biểu mô tuyến đại trực tràng)

---

### 2.3. OrganMNIST - Cụ thể OrganAMNIST / Axial (Cắt lớp vi tính CT ổ bụng)

* **Phương thức ảnh (Modality)**: Ảnh chụp cắt lớp vi tính 3D (3D Abdominal Computed Tomography - CT).
* **Công trình khoa học gốc**:
  1. *Dữ liệu ảnh CT 3D*: Bắt nguồn từ cuộc thi quốc tế **Liver Tumor Segmentation Benchmark (LiTS Challenge)** – Bilic et al., công bố trên **Medical Image Analysis** (2023), 84: 102680 (Preprint arXiv:1901.04056).
  2. *Nhãn phân đoạn 11 cơ quan*: Xu et al., *"Efficient Multiple Organ Localization in CT Image Using 3D Region Proposal Network"*, công bố trên tạp chí hàng đầu **IEEE Transactions on Medical Imaging (IEEE TMI)**, 2019, 38(8): 1885-1895.
* **Cơ sở y tế & Đơn vị thu thập**: 
  - Dữ liệu ảnh CT được tổng hợp từ **6 trung tâm y tế và bệnh viện lớn trên thế giới**:
    1. Bệnh viện Đại học Ludwig Maximilian Munich (LMU), Munich, CHLB Đức.
    2. Trung tâm Y tế Đại học Radboud Nijmegen, Hà Lan.
    3. Viện Phẫu thuật Xâm lấn Tối thiểu IRCAD, Strasbourg, Pháp.
    4. Đại học Tel Aviv / Trung tâm Y tế Chaim Sheba, Israel.
    5. Trung tâm Bệnh viện Đại học Bordeaux (CHU de Bordeaux), Pháp.
    6. Bệnh viện Heilig-Geist Bingen, CHLB Đức.
* **Mức độ Legit & Chuyên gia thẩm định nhãn (Ground Truth Validation)**:
  - Toàn bộ hộp bao định vị 3D (3D Bounding Boxes) và mặt cắt giải phẫu của 11 cơ quan nội tạng đều được gán nhãn và kiểm định độc lập bởi **các Bác sĩ Chẩn đoán Hình ảnh chuyên khoa (Certified Clinical Radiologists)** từ các bệnh viện đối tác thuộc liên minh nghiên cứu LiTS và nhóm tác giả IEEE TMI.
  - Giá trị đơn vị Hounsfield (HU) của CT gốc được chuẩn hóa theo thang cửa sổ ổ bụng tiêu chuẩn (Abdominal Window: `[-160, 240]` HU) chuyển về thang xám `[0, 255]`.
  - Phân tách Train / Val / Test ở **cấp độ thể tích bệnh nhân (Patient-Volume Split)**, bảo đảm các lát cắt của cùng một người bệnh không bao giờ xuất hiện ở cả tập huấn luyện lẫn kiểm thử.
* **Thông số kỹ thuật (OrganAMNIST - Axial)**:
  - Kích thước: `28×28×1` (Grayscale).
  - Số lượng mẫu: **58,850 ảnh** (Train: 34,581 | Val: 6,491 | Test: 17,778).
  - Gồm **11 lớp cơ quan nội tạng**:
    1. Bladder (Bàng quang)
    2. Femur-left (Xương đùi trái)
    3. Femur-right (Xương đùi phải)
    4. Heart (Tim)
    5. Kidney-left (Thận trái)
    6. Kidney-right (Thận phải)
    7. Liver (Gan)
    8. Lung-left (Phổi trái)
    9. Lung-right (Phổi phải)
    10. Pancreas (Tụy)
    11. Spleen (Lách)

---

### 2.4. DermaMNIST (Ảnh soi da liễu / Tổn thương da có sắc tố)

* **Phương thức ảnh (Modality)**: Ảnh soi da kỹ thuật số phóng đại quang học (Dermatoscopic images).
* **Công trình khoa học gốc**: Philipp Tschandl, Christoph Rosendahl, Harald Kittler, *"The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions"*, công bố trên tạp chí danh giá **Nature Scientific Data** (2018), 5: 180161. DOI: `10.1038/sdata.2018.161`.
* **Cơ sở y tế & Đơn vị thu thập**: 
  - **Khoa Da liễu, Đại học Y Vienna (Medical University of Vienna - MedUni Vienna, Áo)** – Đơn vị Nghiên cứu ViDIR Group.
  - **Phòng khám Ung thư Da chuyên sâu của Bác sĩ Cliff Rosendahl tại bang Queensland, Úc** (Trường Y khoa, Đại học Queensland).
* **Mức độ Legit & Quy trình thẩm định 3 tầng (Multi-tier Gold Standard)**:
  - Được chủ trì và giám định trực tiếp bởi các giáo sư, bác sĩ da liễu hàng đầu thế giới về soi da chẩn đoán u hắc tố (Dr. Philipp Tschandl, Prof. Harald Kittler, Dr. Cliff Rosendahl).
  - Nhãn chẩn đoán bệnh được bảo chứng qua **3 cấp độ kiểm định khắt khe**:
    1. **Cấp 1 – Tiêu chuẩn vàng Sinh thiết (Histopathology)**: **Chiếm >53% toàn bộ tập dữ liệu (>5,300 tổn thương), đặc biệt 100% tất cả các ca ung thư ác tính nguy hiểm** (như U hắc tố Melanoma, Ung thư biểu mô tế bào đáy BCC, Ung thư biểu mô tế bào gai / Bowen / AKIEC) đều được chẩn đoán xác định thông qua **sinh thiết giải phẫu bệnh mô học sau phẫu thuật**. Đây là tiêu chuẩn chẩn đoán tuyệt đối trong y khoa.
    2. **Cấp 2 – Đồng thuận Hội đồng Chuyên gia (Expert Consensus)**: Đối với các tổn thương lành tính không cần phẫu thuật cắt bỏ, nhãn được quyết định độc lập bởi hội đồng các bác sĩ da liễu có hơn 10 năm kinh nghiệm soi da lâm sàng.
    3. **Cấp 3 – Theo dõi soi da định kỳ dài hạn (Long-term Follow-up & In-vivo Confocal Microscopy)**: Các tổn thương nevus phẳng được theo dõi chụp ảnh soi da kỹ thuật số định kỳ tối thiểu 1.5 - 2 năm nhằm đảm bảo hình thái hoàn toàn ổn định không có dấu hiệu ác tính hóa, hoặc được kiểm chứng qua kính hiển vi quét đồng tiêu laser in-vivo.
* **Thông số kỹ thuật**:
  - Kích thước: `28×28×3` (RGB).
  - Số lượng mẫu: **10,015 ảnh** (Train: 7,007 | Val: 1,003 | Test: 2,005).
  - Gồm **7 loại bệnh lý da liễu**:
    1. `akiec`: Actinic keratoses and intraepithelial carcinoma (Dày sừng quang hóa & ung thư biểu mô nội biểu mô)
    2. `bcc`: Basal cell carcinoma (Ung thư biểu mô tế bào đáy)
    3. `bkl`: Benign keratosis-like lesions (Tổn thương sừng lành tính: dày sừng tiết bã, đốm nâu solar lentigo)
    4. `df`: Dermatofibroma (U xơ da lành tính)
    5. `mel`: Melanoma (Ung thư hắc tố – u da ác tính nguy hiểm nhất)
    6. `nv`: Melanocytic nevi (Nốt ruồi tế bào hắc tố thông thường)
    7. `vasc`: Vascular lesions (Tổn thương mạch máu ngoài da: u máu, giãn mao mạch)

---

## 3. Bảng Tổng Hợp Đối Sánh Tính Pháp Lý & Mức Độ Legit Y Khoa

| Bộ dữ liệu | Phương thức ảnh (Modality) | Bệnh viện / Cơ sở y tế thu thập | Đội ngũ thẩm định & Phương pháp xác thực nhãn (Ground Truth) | Mức độ Legit (Độ tin cậy lâm sàng) | Quy mô & Số lớp |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BloodMNIST** | Kính hiển vi tế bào máu ngoại vi | **Hospital Clínic de Barcelona (Tây Ban Nha)** (Core Lab) & ĐH Bách khoa Catalonia | Bác sĩ Huyết học lâm sàng & Bác sĩ Giải phẫu bệnh kiểm duyệt trực tiếp qua hệ thống tự động CellaVision DM96. | **Rất cao** (Đạt chuẩn chứng nhận phòng xét nghiệm lâm sàng quốc tế) | 17,092 ảnh<br>*(28×28×3 RGB)*<br>8 lớp tế bào |
| **PathMNIST** | Mô bệnh học H&E cắt lạnh/đúc nến (FFPE) | **Bệnh viện Đại học Heidelberg & TT Ung bướu Quốc gia NCT (Đức)**; DACHS Cohort | Bác sĩ chuyên khoa Giải phẫu bệnh (Senior Pathologists) kiểm tra vi thể từng lát cắt. Test set lấy từ cohort bệnh nhân độc lập hoàn toàn. | **Cực kỳ cao** (Tiêu chuẩn vàng sinh thiết H&E; External cohort validation) | 107,180 ảnh<br>*(28×28×3 RGB)*<br>9 lớp mô học |
| **OrganMNIST** *(OrganAMNIST)* | Cắt lớp vi tính CT ổ bụng 3D | **6 BV & Viện nghiên cứu quốc tế**: LMU Munich (Đức), Radboud (Hà Lan), IRCAD (Pháp), Sheba (Israel), CHU Bordeaux (Pháp)... | Bác sĩ Chẩn đoán hình ảnh (Clinical Radiologists) phân vùng 3D bounding box và kiểm định giải phẫu 11 cơ quan; xuất bản IEEE TMI. | **Rất cao** (Đồng thuận đa trung tâm quốc tế; Patient-volume split) | 58,850 ảnh<br>*(28×28×1 Gray)*<br>11 lớp cơ quan |
| **DermaMNIST** | Ảnh soi da liễu quang học (Dermatoscopy) | **Khoa Da liễu, ĐH Y Vienna (Áo)** (ViDIR) & PK Ung thư da Queensland (Úc) | >53% ca bệnh (và 100% ca ác tính: Melanoma, BCC) có sinh thiết mô bệnh học xác nhận; phần còn lại qua hội đồng chuyên gia & theo dõi lâm sàng 2 năm. | **Cực kỳ cao** (Tiêu chuẩn vàng sinh thiết cho 100% u ác tính; Nature Sci Data) | 10,015 ảnh<br>*(28×28×3 RGB)*<br>7 lớp tổn thương |

---

## 4. Ý Nghĩa Của Mức Độ "Legit" Này Đối Với Nghiên Cứu Federated Learning

1. **Loại trừ tuyệt đối yếu tố nhiễu nhãn (Clean Ground Truth)**:
   - Trong Federated Learning, hiện tượng dữ liệu không đồng nhất (Non-IID) giữa các client gây ra hiện tượng trôi dạt trọng số (*Client Drift*).
   - Nếu nhãn của dữ liệu bị sai (Label Noise do gán nhãn tự động không kiểm duyệt hoặc gắn nhãn ẩu), việc đánh giá hiệu năng thuật toán sẽ bị sai lệch hoàn toàn — người làm nghiên cứu sẽ không thể xác định được nguyên nhân suy giảm độ chính xác là do thuật toán FL (FedAvg, FedProx, FedSLD...) hay do nhãn dữ liệu bị lỗi.
   - Với việc cả 4 bộ dữ liệu trên đều có **Ground Truth được thẩm định trực tiếp bởi Bác sĩ chuyên khoa / Tiêu chuẩn vàng sinh thiết**, nghiên cứu của chúng ta đảm bảo 100% sai số đo lường phản ánh đúng bản chất của giải thuật phân tán.

2. **Mô phỏng chân thực kịch bản Đa Bệnh viện Độc lập**:
   - Việc chia tách train/test ở **cấp độ bệnh nhân (Patient-Level Split)** và có các tập test ngoại suy độc lập (như nhóm bệnh nhân Heidelberg vs DACHS trong PathMNIST, hoặc 6 bệnh viện trong LiTS/OrganMNIST) giúp mô phỏng chính xác kịch bản thực tế khi các bệnh viện khác nhau huấn luyện mô hình mà không làm rò rỉ dữ liệu bệnh nhân của nhau.

3. **Cơ sở bảo vệ khoa học vững chắc**:
   - Khi bảo vệ trước Hội đồng Khoa học, phản biện bài báo hoặc xuất bản quốc tế, nguồn gốc dữ liệu từ các tổ chức uy tín hàng đầu thế giới (Heidelberg, Barcelona, MedUni Vienna, LiTS Consortium) cùng các bài báo công bố trên Nature Scientific Data, PLOS Medicine, IEEE TMI là tấm bằng bảo chứng vững chắc nhất về tính hợp lệ và mức độ tin cậy của đề tài FedMedAI.
