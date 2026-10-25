"""
generate_dataset_info_docx.py
Script tự động tạo file Word: dataset_information.docx
Chứa toàn bộ thông tin chi tiết, xuất xứ y khoa, và mức độ kiểm định (legitimacy)
của các bộ dữ liệu MedMNIST: BloodMNIST, PathMNIST, OrganMNIST, DermaMNIST.
"""

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    """Đặt màu nền cho cell trong bảng."""
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=120, bottom=120, left=180, right=180):
    """Đặt lề trong cell (đơn vị dxa, 20 dxa = 1 pt)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table, color="D3D3D3", sz="4", val="single"):
    """Đặt border xám mảnh, trang nhã cho toàn bộ bảng."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:right w:val="none"/>'
        f'  <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def add_callout_box(doc, text_list, bg_color="F0F4F8", border_color="1F497D"):
    """Thêm một khung ghi chú nổi bật (callout box)."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.7)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    # Border trái đậm, các cạnh khác none
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'  <w:top w:val="none"/>'
        f'  <w:left w:val="single" w:sz="24" w:space="0" w:color="{border_color}"/>'
        f'  <w:bottom w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    
    for i, line_info in enumerate(text_list):
        if i > 0:
            p = cell.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.15
        
        # line_info: tuple (prefix_bold, normal_text, is_italic)
        prefix, content, italic = line_info
        if prefix:
            r_pre = p.add_run(prefix)
            r_pre.bold = True
            r_pre.font.name = "Calibri"
            r_pre.font.size = Pt(10.5)
            r_pre.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
        
        r_txt = p.add_run(content)
        r_txt.font.name = "Calibri"
        r_txt.font.size = Pt(10)
        r_txt.italic = italic
        r_txt.font.color.rgb = RGBColor(0x2A, 0x2A, 0x2A)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def build_dataset_information_doc():
    doc = docx.Document()
    
    # Cấu hình lề trang tiêu chuẩn A4 / Letter
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)
        
    # Tiêu đề tài liệu
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(4)
    title_p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = title_p.add_run("BÁO CÁO THẨM ĐỊNH NGUỒN GỐC & TÍNH XÁC THỰC LÂM SÀNG CỦA CÁC BỘ DỮ LIỆU MEDMNIST")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(16)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    
    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(14)
    sub_p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = sub_p.add_run("Tài Liệu Căn Cứ Khoa Học: Xuất Xứ Bệnh Viện, Phương Pháp Xác Thực Nhãn & Mức Độ Legit\n(OrganMNIST, PathMNIST, BloodMNIST, DermaMNIST)")
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(11.5)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(0x59, 0x59, 0x59)

    # ─────────────────────────────────────────────────────────────
    # MỤC 1: Tổng quan MedMNIST Benchmark
    # ─────────────────────────────────────────────────────────────
    h1 = doc.add_heading(level=1)
    r1 = h1.add_run("1. Tổng quan về Chuẩn Benchmark MedMNIST")
    r1.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    r1.font.name = "Calibri"

    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)
    p.add_run("Bộ dữ liệu mà dự án FedMedAI đang sử dụng thuộc hệ sinh thái ")
    r_bold = p.add_run("MedMNIST v2 (Large-Scale Lightweight Benchmark for Biomedical Image Classification)")
    r_bold.bold = True
    p.add_run(", được công bố chính thức trên tạp chí hàng đầu ")
    r_nat = p.add_run("Nature Scientific Data (Volume 10, Article 41, 2023)")
    r_nat.bold = True
    p.add_run(" bởi nhóm tác giả Jiancheng Yang, Rui Shi, Donglai Wei, Hanspeter Pfister (Đại học Harvard, Đại học Giao thông Thượng Hải, v.v.).")

    add_callout_box(doc, [
        ("Mục tiêu thiết kế: ", "Chuẩn hóa các phương thức ảnh y tế đa dạng (CT, Nội soi da, Tiêu bản mô học, Kính hiển vi máu...) về kích thước chuẩn hóa 28×28 pixels (hoặc 28×28×28 cho 3D) tương tự MNIST, loại bỏ rào cản tính toán mà vẫn giữ nguyên bản chất hình thái học y tế.", False),
        ("Quy mô: ", "Gồm 18 bộ dữ liệu y khoa (12 tập 2D và 6 tập 3D) với tổng cộng hơn 708,000 ảnh 2D và gần 10,000 khối thể tích 3D.", False),
        ("Chia tập chuẩn (Standard Split): ", "Tất cả các bộ dữ liệu đều được phân tách cố định thành Train / Validation / Test ở cấp độ bệnh nhân (Patient-Level Split), đảm bảo tuyệt đối không có hiện tượng rò rỉ dữ liệu (Data Leakage) giữa các tập.", False),
        ("Tính hợp thức cho Federated Learning: ", "Mỗi ảnh đều sở hữu nhãn gốc đã được chuyên gia/bác sĩ thẩm định lâm sàng (Clean Ground Truth). Nhờ đó, bất kỳ sự thay đổi nào về độ chính xác giữa các thuật toán FL (FedAvg, FedProx, FedSLD...) đều phản ánh chính xác năng lực xử lý Non-IID của thuật toán chứ không bị nhiễu do nhãn sai.", False)
    ])

    # ─────────────────────────────────────────────────────────────
    # MỤC 2: Chi tiết 4 bộ dữ liệu nghiên cứu
    # ─────────────────────────────────────────────────────────────
    h2 = doc.add_heading(level=1)
    r2 = h2.add_run("2. Nguồn gốc, Xuất xứ Bệnh viện & Thẩm định Chuyên gia cho 4 Bộ Dữ liệu")
    r2.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    r2.font.name = "Calibri"

    # --- 2.1 BloodMNIST ---
    h2_1 = doc.add_heading(level=2)
    r2_1 = h2_1.add_run("2.1. BloodMNIST (Kính hiển vi Tế bào Máu Ngoại vi)")
    r2_1.font.color.rgb = RGBColor(0x2E, 0x75, 0xB6)
    r2_1.font.name = "Calibri"

    add_callout_box(doc, [
        ("Cơ sở y tế & Địa điểm: ", "Core Laboratory, Bệnh viện Đại học Barcelona (Hospital Clínic de Barcelona), Tây Ban Nha phối hợp cùng Đại học Bách khoa Catalonia (UPC BarcelonaTech).", False),
        ("Công trình khoa học gốc: ", "Acevedo et al. (2020), \"A dataset of microscopic peripheral blood cell images for development of automatic recognition systems\", công bố trên tạp chí quốc tế Data in Brief, Elsevier (DOI: 10.1016/j.dib.2020.105474).", True),
        ("Đội ngũ chuyên gia thẩm định (Mức độ Legit): ", "100% hình ảnh tế bào đơn lẻ được soi, phân loại và xác thực trực tiếp bởi các Bác sĩ chuyên khoa Huyết học lâm sàng (Certified Clinical Hematologists) và Bác sĩ Giải phẫu bệnh tại Core Laboratory của Bệnh viện Hospital Clínic de Barcelona.", False),
        ("Thiết bị thu thập: ", "Hệ thống chụp và phân tích tế bào tự động cao cấp CellaVision DM96 – thiết bị chuẩn vàng đạt chứng nhận lâm sàng quốc tế trong các phòng xét nghiệm huyết học.", False),
        ("Thông số kỹ thuật: ", "Tổng cộng 17,092 ảnh RGB (Train: 11,959 | Val: 1,712 | Test: 3,421), kích thước chuẩn 28×28×3. Gồm 8 lớp tế bào máu ngoại vi thu từ 100 bệnh nhân thường quy không mắc bệnh ác tính: Basophils, Eosinophils, Erythroblasts, Immature granulocytes, Lymphocytes, Monocytes, Neutrophils, Platelets.", False)
    ])

    # --- 2.2 PathMNIST ---
    h2_2 = doc.add_heading(level=2)
    r2_2 = h2_2.add_run("2.2. PathMNIST (Mô bệnh học Ung thư Đại Trực tràng - Colorectal Cancer Histology)")
    r2_2.font.color.rgb = RGBColor(0x2E, 0x75, 0xB6)
    r2_2.font.name = "Calibri"

    add_callout_box(doc, [
        ("Cơ sở y tế & Địa điểm: ", "Trung tâm Quốc gia về Bệnh lý Khối u (National Center for Tumor Diseases - NCT), Bệnh viện Đại học Heidelberg (Heidelberg University Hospital), Đức, phối hợp cùng Trung tâm Nghiên cứu Ung thư Đức (DKFZ) và Trung tâm Y tế Đại học Mannheim.", False),
        ("Công trình khoa học gốc: ", "Jakob Nikolas Kather et al. (2019), \"Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study\", công bố trên PLOS Medicine (DOI: 10.1371/journal.pmed.1002730) và nghiên cứu liên quan trên Nature Medicine (2019).", True),
        ("Đội ngũ chuyên gia thẩm định (Mức độ Legit): ", "Tiêu bản mô bệnh học được cắt từ khối mô đúc sáp parafin cố định formalin (FFPE) từ các ca phẫu thuật ung thư đại trực tràng, nhuộm Hematoxylin & Eosin (H&E). Các bác sĩ giải phẫu bệnh chuyên sâu về ung thư (Certified Senior Pathologists) tại Viện Giải phẫu bệnh Bệnh viện Đại học Heidelberg đã trực tiếp kiểm tra thủ công từng vi trường mô học.", False),
        ("Đặc tính bảo chứng ngoại suy (Zero Patient Overlap): ", "Tập Test (7,180 ảnh) bắt nguồn từ tập dữ liệu độc lập CRC-VAL-HE-7K thuộc nhóm thuần tập nghiên cứu DACHS (50 bệnh nhân mới hoàn toàn), tuyệt đối KHÔNG có bất kỳ bệnh nhân nào trùng lặp với tập Train/Val (lấy từ NCT-CRC-HE-100K gồm 86 bệnh nhân). Đây là tiêu chuẩn kiểm định khắt khe nhất trong y học.", False),
        ("Thông số kỹ thuật: ", "Tổng cộng 107,180 ảnh RGB (Train: 89,996 | Val: 10,004 | Test: 7,180), kích thước 28×28×3. Gồm 9 lớp cấu trúc mô: (1) Mô mỡ Adipose, (2) Nền tiêu bản Background, (3) Mảnh vụn tế bào Debris, (4) Tế bào Lympho, (5) Chất nhầy Mucus, (6) Cơ trơn Smooth muscle, (7) Niêm mạc đại tràng lành tính Normal mucosa, (8) Mô đệm liên kết khối u Stroma, (9) Biểu mô ung thư biểu mô tuyến đại trực tràng Tumor epithelium.", False)
    ])

    # --- 2.3 OrganMNIST ---
    h2_3 = doc.add_heading(level=2)
    r2_3 = h2_3.add_run("2.3. OrganMNIST - Cụ thể OrganAMNIST / Axial (Chụp cắt lớp Vi tính CT Ổ bụng)")
    r2_3.font.color.rgb = RGBColor(0x2E, 0x75, 0xB6)
    r2_3.font.name = "Calibri"

    add_callout_box(doc, [
        ("Cơ sở y tế & Địa điểm: ", "Tập dữ liệu đa trung tâm quốc tế (Multi-center International Cohort) thu thập từ 6 bệnh viện & viện nghiên cứu hàng đầu:\n  • Bệnh viện Đại học Ludwig Maximilian Munich (LMU), Đức\n  • Trung tâm Y tế Đại học Radboud Nijmegen, Hà Lan\n  • Viện Phẫu thuật Xâm lấn Tối thiểu IRCAD, Strasbourg, Pháp\n  • Đại học Tel Aviv / Trung tâm Y tế Sheba, Israel\n  • Bệnh viện Đại học Bordeaux (CHU de Bordeaux), Pháp\n  • Bệnh viện Đa khoa Heilig-Geist Bingen, Đức.", False),
        ("Công trình khoa học gốc: ", "1) Ảnh CT 3D gốc từ Liver Tumor Segmentation Benchmark (LiTS Challenge) - Bilic et al., công bố trên Medical Image Analysis (2023) / arXiv:1901.04056.\n2) Nhãn định vị 11 cơ quan: Nghiên cứu của Xu et al. (2019), \"Efficient Multiple Organ Localization in CT Image Using 3D Region Proposal Network\", công bố trên tạp chí IEEE Transactions on Medical Imaging (IEEE TMI, DOI: 10.1109/TMI.2018.2882081).", True),
        ("Đội ngũ chuyên gia thẩm định (Mức độ Legit): ", "Toàn bộ hộp bao 3D (3D Bounding Boxes) và đường viền phân vùng giải phẫu của 11 cơ quan nội tạng đều được thiết lập, kiểm duyệt độc lập và chuẩn hóa bởi các Bác sĩ Chẩn đoán Hình ảnh (Certified Clinical Radiologists) tại các bệnh viện đối tác thuộc liên minh nghiên cứu.", False),
        ("Quy trình xử lý ảnh y khoa: ", "Đơn vị Hounsfield Units (HU) từ ảnh CT nguyên bản được chuẩn hóa qua cửa sổ mô mềm ổ bụng (Abdominal Window: [-160, 240] HU) về thang xám [0, 255]. Lát cắt 2D được trích xuất từ tâm của các hộp bao 3D theo mặt phẳng Axial (mặt phẳng cắt ngang - OrganAMNIST). Việc chia tập train/val/test được thực hiện ở cấp độ từng bệnh nhân (Patient-Volume Split).", False),
        ("Thông số kỹ thuật: ", "Tổng cộng 58,850 ảnh đơn kênh Grayscale (Train: 34,581 | Val: 6,491 | Test: 17,778), kích thước 28×28×1. Gồm 11 cơ quan: Bàng quang (Bladder), Xương đùi trái (Femur-L), Xương đùi phải (Femur-R), Tim (Heart), Thận trái (Kidney-L), Thận phải (Kidney-R), Gan (Liver), Phổi trái (Lung-L), Phổi phải (Lung-R), Tụy (Pancreas), Lách (Spleen).", False)
    ])

    # --- 2.4 DermaMNIST ---
    h2_4 = doc.add_heading(level=2)
    r2_4 = h2_4.add_run("2.4. DermaMNIST (Ảnh Soi Da Liễu - Dermatoscopy / Skin Lesions)")
    r2_4.font.color.rgb = RGBColor(0x2E, 0x75, 0xB6)
    r2_4.font.name = "Calibri"

    add_callout_box(doc, [
        ("Cơ sở y tế & Địa điểm: ", "1) Khoa Da liễu, Đại học Y Vienna (Medical University of Vienna), Áo - Đơn vị nghiên cứu ViDIR Group.\n2) Phòng khám Ung thư Da chuyên sâu của Bác sĩ Cliff Rosendahl tại Queensland, Úc (Trường Y khoa, Đại học Queensland).", False),
        ("Công trình khoa học gốc: ", "Philipp Tschandl, Christoph Rosendahl, Harald Kittler (2018), \"The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions\", công bố trên tạp chí danh giá Nature Scientific Data (5:180161, DOI: 10.1038/sdata.2018.161).", True),
        ("Đội ngũ chuyên gia thẩm định & Tiêu chuẩn vàng (Mức độ Legit): ", "Dữ liệu được bảo chứng bởi quy trình kiểm định 3 tầng cực kỳ khắt khe của hội đồng chuyên gia da liễu hàng đầu:\n  • Tầng 1 - Tiêu chuẩn vàng Sinh thiết (Histopathology): Hơn 53% tổng số mẫu (>5,300 ca), đặc biệt 100% tất cả các tổn thương ác tính nguy hiểm (Melanoma, Basal Cell Carcinoma, Bowen's disease / AKIEC) đều được chẩn đoán xác định qua sinh thiết mô bệnh học sau phẫu thuật.\n  • Tầng 2 - Đồng thuận Chuyên gia (Expert Consensus): Đối với các tổn thương sắc tố lành tính không có chỉ định cắt bỏ, nhãn được quyết định bởi sự đồng thuận độc lập của các bác sĩ chuyên khoa da liễu có trên 10 năm kinh nghiệm soi da lâm sàng.\n  • Tầng 3 - Theo dõi lâm sàng dài hạn (Long-term Digital Dermatoscopy Follow-up): Các tổn thương nevus phẳng được theo dõi chụp ảnh định kỳ tối thiểu 1.5 - 2 năm không có biến đổi kích thước hoặc soi qua kính hiển vi quét đồng tiêu in-vivo.", False),
        ("Thông số kỹ thuật: ", "Tổng cộng 10,015 ảnh RGB (Train: 7,007 | Val: 1,003 | Test: 2,005), kích thước 28×28×3. Gồm 7 loại bệnh học da liễu quan trọng:\n  1. akiec: Dày sừng quang hóa & ung thư biểu mô nội biểu mô (Actinic keratoses)\n  2. bcc: Ung thư biểu mô tế bào đáy (Basal cell carcinoma)\n  3. bkl: Tổn thương sừng lành tính (Benign keratosis-like lesions)\n  4. df: U xơ da lành tính (Dermatofibroma)\n  5. mel: Ung thư hắc tố (Melanoma - u da ác tính nguy hiểm nhất)\n  6. nv: Nốt ruồi tế bào hắc tố thông thường (Melanocytic nevi)\n  7. vasc: Tổn thương mạch máu da (Vascular lesions).", False)
    ])

    # ─────────────────────────────────────────────────────────────
    # MỤC 3: Bảng Tổng Hợp So Sánh Đối Chiếu
    # ─────────────────────────────────────────────────────────────
    h3 = doc.add_heading(level=1)
    r3 = h3.add_run("3. Bảng Tổng Hợp Đối Sánh Tính Pháp Lý & Xác Thực Y Khoa")
    r3.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    r3.font.name = "Calibri"

    table_data = [
        ["Bộ dữ liệu", "Phương thức (Modality)", "Bệnh viện / Cơ sở y tế thu thập", "Chuyên gia / Phương pháp thẩm định nhãn (Ground Truth)", "Độ tin cậy (Legitimacy)", "Kích thước & Lớp"],
        [
            "BloodMNIST",
            "Kính hiển vi tế bào máu (Microscopic Smear)",
            "Hospital Clínic de Barcelona, Tây Ban Nha (Core Lab) & UPC",
            "Bác sĩ Huyết học lâm sàng & Bác sĩ Giải phẫu bệnh kiểm duyệt trực tiếp qua hệ thống CellaVision DM96.",
            "Rất cao (Chứng chỉ lâm sàng phòng xét nghiệm chuẩn quốc tế)",
            "17,092 ảnh\n(28×28×3 RGB)\n8 lớp tế bào"
        ],
        [
            "PathMNIST",
            "Giải phẫu bệnh mô học H&E (Histopathology)",
            "Trung tâm Ung bướu Quốc gia NCT & BV Đại học Heidelberg (Đức); DACHS Study",
            "Bác sĩ chuyên khoa Giải phẫu bệnh ung thư (Pathologists) kiểm tra vi thể từng lát cắt FFPE. Test set từ cohort bệnh nhân độc lập hoàn toàn.",
            "Cực kỳ cao (Tiêu chuẩn vàng sinh thiết H&E; External cohort validation)",
            "107,180 ảnh\n(28×28×3 RGB)\n9 lớp mô học"
        ],
        [
            "OrganMNIST\n(OrganAMNIST)",
            "Chụp cắt lớp vi tính CT ổ bụng (Abdominal CT)",
            "Đa trung tâm quốc tế (6 bệnh viện: LMU Munich, Radboud Hà Lan, IRCAD Pháp, Sheba Israel, CHU Bordeaux...)",
            "Bác sĩ Chẩn đoán hình ảnh (Radiologists) phân vùng 3D bounding box và kiểm định giải phẫu 11 cơ quan; xuất bản IEEE TMI.",
            "Rất cao (Đồng thuận đa trung tâm quốc tế; Patient-volume split)",
            "58,850 ảnh\n(28×28×1 Gray)\n11 lớp cơ quan"
        ],
        [
            "DermaMNIST",
            "Ảnh soi da liễu quang học (Dermatoscopy)",
            "Đại học Y Vienna (Áo) & ViDIR Group; PK Ung thư da Queensland (Úc)",
            ">53% ca (và 100% ca ác tính: Melanoma, BCC) có sinh thiết mô bệnh học xác nhận; phần còn lại qua hội đồng chuyên gia & theo dõi 2 năm.",
            "Cực kỳ cao (Tiêu chuẩn vàng sinh thiết cho 100% u ác tính; Nature Sci Data)",
            "10,015 ảnh\n(28×28×3 RGB)\n7 lớp bệnh da"
        ]
    ]

    col_widths = [Inches(1.0), Inches(1.1), Inches(1.4), Inches(1.7), Inches(1.1), Inches(0.9)]
    
    tab = doc.add_table(rows=len(table_data), cols=len(table_data[0]))
    tab.style = 'Table Grid'
    tab.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tab, color="B0C4DE", sz="4")

    for r_idx, row in enumerate(tab.rows):
        for c_idx, cell in enumerate(row.cells):
            cell.width = col_widths[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
            
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.05
            
            text_val = table_data[r_idx][c_idx]
            
            if r_idx == 0:
                set_cell_background(cell, "1F497D")
                r_txt = p.add_run(text_val)
                r_txt.font.name = "Calibri"
                r_txt.font.size = Pt(9.5)
                r_txt.bold = True
                r_txt.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                if r_idx % 2 == 1:
                    set_cell_background(cell, "FFFFFF")
                else:
                    set_cell_background(cell, "F7F9FB")
                
                r_txt = p.add_run(text_val)
                r_txt.font.name = "Calibri"
                r_txt.font.size = Pt(8.5)
                r_txt.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
                if c_idx == 0:
                    r_txt.bold = True
                    r_txt.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)

    p_space = doc.add_paragraph()
    p_space.paragraph_format.space_after = Pt(8)

    # ─────────────────────────────────────────────────────────────
    # MỤC 4: Ý nghĩa đối với Federated Learning (FedMedAI)
    # ─────────────────────────────────────────────────────────────
    h4 = doc.add_heading(level=1)
    r4 = h4.add_run("4. Ý Nghĩa Của Tính Xác Thực Nhãn Đối Với Nghiên Cứu Federated Learning")
    r4.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    r4.font.name = "Calibri"

    add_callout_box(doc, [
        ("1. Loại bỏ biến gây nhiễu (Confounding Factor) do sai nhãn: ", "Trong học liên kết (FL), phân phối dữ liệu không đồng nhất (Non-IID) giữa các bệnh viện vốn dĩ gây ra độ lệch trọng số (Client Drift). Nếu nhãn dữ liệu bị gán ẩu bởi mô hình tự động hoặc người không có chuyên môn, sai số hội tụ của thuật toán sẽ bị hòa lẫn với nhiễu nhãn (Label Noise). Với 4 bộ dữ liệu này, độ tin cậy nhãn ở mức lâm sàng cao nhất, bảo đảm mọi kết luận so sánh giữa FedAvg, FedProx hay FedSLD đều phản ánh năng lực giải thuật thuần túy.", False),
        ("2. Mô phỏng chính xác kịch bản Bệnh viện Độc lập: ", "Các phân chia train/test theo cấp độ bệnh nhân (Patient-Level) và các cơ sở y tế độc lập (như nhóm bệnh nhân NCT vs DACHS trong PathMNIST, hoặc 6 bệnh viện trong LiTS/OrganMNIST) cung cấp nền tảng hoàn hảo để kiểm thử tính tổng quát hóa (Generalization) của mô hình FL trên dữ liệu chưa từng thấy.", False),
        ("3. Giá trị thuyết phục khi Bảo vệ Đề tài & Xuất bản Khoa học: ", "Cả 4 bộ dữ liệu đều xuất phát từ các công trình danh giá (Nature Scientific Data, PLOS Medicine, IEEE TMI, Data in Brief) và các bệnh viện hàng đầu thế giới (Heidelberg, Barcelona, MedUni Vienna, LiTS Consortium). Điều này bảo đảm tính pháp lý và cơ sở khoa học vững chắc trước bất kỳ Hội đồng Khoa học hay phản biện chuyên môn nào.", False)
    ], bg_color="EBF3FB", border_color="005B96")

    output_path = "dataset_information.docx"
    doc.save(output_path)
    print("Created successfully: " + output_path)

if __name__ == "__main__":
    build_dataset_information_doc()
