"""
generate_review2_docx.py
Tạo file review_2.docx:
Tổng hợp súc tích, đắt giá, cấu trúc slide-ready về:
1. Phân loại đa chiều hệ thống FL (Topology, Scale, Non-IID taxonomy từ Kairouz et al. 2021)
2. Nhóm thuật toán (Client Drift vs Personalized FL / Feature Shift)
3. So sánh nền tảng phần mềm & Lý do lựa chọn Flower (flwr)
"""

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table, color="D3D3D3", sz="4", val="single"):
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

def add_slide_card(doc, title, bullet_points, bg_color="F2F5F9", border_color="1F497D"):
    """Thêm một khung dạng Slide Card ngắn gọn, súc tích."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.8)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
    
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
    run_t = p.add_run(f"📌 {title.upper()}")
    run_t.bold = True
    run_t.font.name = "Calibri"
    run_t.font.size = Pt(11)
    run_t.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    
    for bp in bullet_points:
        p_bp = cell.add_paragraph()
        p_bp.paragraph_format.space_before = Pt(1)
        p_bp.paragraph_format.space_after = Pt(2)
        p_bp.paragraph_format.line_spacing = 1.12
        
        prefix, content = bp
        r_pre = p_bp.add_run(f"• {prefix}: " if prefix else "• ")
        r_pre.bold = True
        r_pre.font.name = "Calibri"
        r_pre.font.size = Pt(10)
        r_pre.font.color.rgb = RGBColor(0x2B, 0x54, 0x7E) if prefix else RGBColor(0x22, 0x22, 0x22)
        
        r_con = p_bp.add_run(content)
        r_con.font.name = "Calibri"
        r_con.font.size = Pt(9.5)
        r_con.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
        
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def build_review_2_docx():
    doc = docx.Document()
    
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)
        
    # Tiêu đề
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(3)
    title_p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = title_p.add_run("REVIEW 2: PHÂN LOẠI HỆ THỐNG FEDERATED LEARNING & LỰA CHỌN NỀN TẢNG")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(15.5)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    
    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(12)
    sub_p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = sub_p.add_run("Tài Liệu Tổng Hợp Trình Bày Slide (Slide-Ready Notes)\nTham chiếu trực tiếp: \"Advances and Open Problems in Federated Learning\" (Kairouz et al., 2021)")
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(11)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(0x59, 0x59, 0x59)

    # ─────────────────────────────────────────────────────────────
    # PHẦN 1: PHÂN LOẠI ĐA CHIỀU CÁC HỆ THỐNG FL HIỆN CÓ
    # ─────────────────────────────────────────────────────────────
    h1 = doc.add_heading(level=1)
    r1 = h1.add_run("1. Phân Loại Đa Chiều Các Hệ Thống Federated Learning")
    r1.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    r1.font.name = "Calibri"

    # 1.1 Architecture & Topology
    add_slide_card(doc, "1.1. Architecture & Topology: Centralized FL vs. Decentralized FL", [
        ("Centralized FL (Server–Client / Hub-and-Spoke)", "Có máy chủ trung tâm (Parameter Server) điều phối: chọn client, phân phối mô hình toàn cục, tổng hợp trọng số (Aggregation). Ưu: Hội tụ nhanh, dễ kiểm soát, chuẩn hóa cao. Nhược: Nghẽn băng thông server (Bandwidth Bottleneck) và điểm chết đơn độc (Single Point of Failure)."),
        ("Decentralized FL (Peer-to-Peer / Gossip)", "Không có máy chủ trung tâm. Các client trao đổi tham số cục bộ với láng giềng theo cấu trúc đồ thị mạng (Graph Topology). Ưu: Loại bỏ điểm chết đơn độc, tăng tính tự trị. Nhược: Tốc độ hội tụ chậm, chi phí truyền thông topo phức tạp, khó kiểm soát chất lượng mô hình."),
        ("Lựa chọn cho Y tế", "Đa số các hệ thống FL y tế thực tế áp dụng Centralized FL có bảo mật (Secure Aggregation) nhờ tính quản trị tập trung và phù hợp quy chuẩn quản lý bệnh viện.")
    ])

    # 1.2 Deployment Scale: Cross-Silo vs Cross-Device
    add_slide_card(doc, "1.2. Quy Mô Triển Khai: Cross-Silo FL vs. Cross-Device FL (Table 1 - Kairouz et al.)", [
        ("Cross-Silo FL (Học liên kết giữa các Tổ chức/Bệnh viện)", "Quy mô: 2 – 100 clients (bệnh viện, trung tâm y tế, datacenter). Phần cứng: Máy trạm/Server/GPU mạnh. Mạng: Băng thông rộng, ổn định. Tính sẵn sàng: Gần 100% client online liên tục mỗi round (Stateful). Dữ liệu: Phân vùng theo ví dụ (Horizontal) hoặc theo đặc trưng (Vertical)."),
        ("Cross-Device / Edge FL (Thiết bị Biên / Di động / IoT)", "Quy mô: Hàng triệu thiết bị (Mobile, IoT, smartwatch). Phần cứng: Giới hạn (RAM, pin yếu). Mạng: Không ổn định (WiFi/4G). Tính sẵn sàng: <1% client được chọn mỗi round, tỷ lệ drop-out cao (Stateless)."),
        ("Vị trí của Dự án FedMedAI", "Dự án tập trung vào Cross-Silo Medical FL (các bệnh viện/khoa lâm sàng phối hợp huấn luyện) nhưng có khả năng mở rộng xuống Edge Medical FL (các máy trạm y tế nhúng Edge AI / Jetson).")
    ])

    # Bảng 1: So sánh Cross-Silo vs Cross-Device
    t_scale = doc.add_table(rows=6, cols=3)
    t_scale.style = 'Table Grid'
    t_scale.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_scale, color="B0C4DE")

    scale_headers = ["Tiêu Chí So Sánh", "Cross-Silo FL (Đồ án áp dụng)", "Cross-Device / Edge FL"]
    scale_rows = [
        ["Đối tượng tham gia (Clients)", "2 – 100 Bệnh viện / Trung tâm nghiên cứu", "10^3 – 10^10 Thiết bị di động / IoT"],
        ["Tài nguyên tính toán (Compute)", "Máy chủ, GPU y tế chuyên dụng, bộ nhớ lớn", "Thiết bị di động, chip biên nhúng (giới hạn pin/RAM)"],
        ["Tính khả dụng & Trạng thái", "Gần 100% online, ổn định (Stateful clients)", "<1% online/round, drop-out ngẫu nhiên (Stateless)"],
        ["Nghẽn cổ chai chính", "Chi phí tính toán hoặc kiểm định pháp lý", "Băng thông mạng truyền thông (Communication bottleneck)"],
        ["Mục tiêu bài toán", "Huấn luyện mô hình chẩn đoán y khoa chính xác cao", "Dự đoán từ tiếp theo, gợi ý ứng dụng người dùng"]
    ]

    for c_i, h_text in enumerate(scale_headers):
        c = t_scale.rows[0].cells[c_i]
        c.text = h_text
        set_cell_background(c, "1F497D")
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = "Calibri"
            r.font.size = Pt(9.5)
            r.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    for r_i, r_data in enumerate(scale_rows):
        row = t_scale.rows[r_i + 1]
        for c_i, val in enumerate(r_data):
            cell = row.cells[c_i]
            cell.text = val
            set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
            if r_i % 2 == 1:
                set_cell_background(cell, "F7F9FB")
            p = cell.paragraphs[0]
            for r in p.runs:
                r.font.name = "Calibri"
                r.font.size = Pt(8.5)
                if c_i == 0:
                    r.bold = True
                    r.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 1.3 Non-IID Taxonomy (Không dùng ký hiệu toán học phức tạp, trực quan cho slide)
    add_slide_card(doc, "1.3. Phân Vùng Dữ Liệu: Các Dạng Lệch Dữ Liệu Non-IID Trong Y Tế", [
        ("1. Lệch Tỷ Lệ Bệnh Giữa Các Viện (Label Distribution Skew)", 
         "Tỷ lệ các nhóm bệnh rất khác nhau giữa các bệnh viện. Ví dụ: Bệnh viện Ung bướu tiếp nhận đa số bệnh nhân ung thư, trong khi Bệnh viện Đa khoa chủ yếu tiếp nhận bệnh nhân thông thường và rất ít ca ung thư."),
        ("2. Lệch Đặc Trưng Ảnh Do Thiết Bị Chụp (Feature Distribution Skew / Scanner Bias)", 
         "Cùng một loại bệnh nhưng hình ảnh chụp ra có độ sáng, độ tương phản và độ nhiễu khác nhau do các bệnh viện dùng máy chụp của các hãng khác nhau (Siemens, GE, Philips)."),
        ("3. Lệch Tiêu Chuẩn Chẩn Đoán & Biểu Hiện Bệnh (Concept Drift)", 
         "• Cùng bệnh nhưng biểu hiện khác nhau: Cùng tổn thương phổi nhưng hình ảnh X-quang ở người cao tuổi khác nhiều so với trẻ em.\n"
         "• Cùng hình ảnh nhưng chẩn đoán khác nhau: Cùng một lát cắt mô bệnh học nhưng ranh giới phân loại khối u giữa các trường phái y tế có độ vênh nhất định."),
        ("4. Lệch Số Lượng Ca Bệnh (Quantity Skew)", 
         "Quy mô dữ liệu chênh lệch rất lớn: Bệnh viện tuyến Trung ương có hàng trăm nghìn ca bệnh, trong khi phòng khám cơ sở chỉ có vài chục ca."),
        ("5. Phân Vùng Dữ Liệu Thực Tế Trong Đồ Án FedMedAI", 
         "• Bộ dữ liệu sử dụng: Thực nghiệm trọng tâm trên OrganMNIST-Axial (11 nhóm cơ quan CT ổ bụng, 58 nghìn ảnh); mở rộng và đối chiếu với BloodMNIST (8 nhóm tế bào máu) và DermaMNIST (7 nhóm bệnh da).\n"
         "• Dạng lệch dữ liệu trọng tâm: Tập trung giải quyết Lệch Tỷ Lệ Bệnh Giữa Các Viện (Label Distribution Skew) — bài toán cốt lõi mà phương pháp đề xuất FedSLD (chia sẻ phân phối nhãn và hiệu chỉnh phân loại) giải quyết triệt để.\n"
         "• Kịch bản thực nghiệm (12 bệnh viện / clients):\n"
         "   - Kịch bản Thực tế (Practical Non-IID): Mô phỏng các bệnh viện đa khoa cùng tiếp nhận nhiều loại bệnh nhưng với tỷ lệ ca bệnh chênh lệch lớn giữa các viện.\n"
         "   - Kịch bản Chuyên khoa Cực đoan (Pathological Non-IID): Mỗi bệnh viện chỉ tiếp nhận 1 đến 2 nhóm bệnh chuyên sâu, tạo thử thách tối đa để kiểm tra độ ổn định của thuật toán.")
    ])

    # ─────────────────────────────────────────────────────────────
    # PHẦN 2: NHÓM HỆ THỐNG THUẬT TOÁN & TỔNG HỢP
    # ─────────────────────────────────────────────────────────────
    h2 = doc.add_heading(level=1)
    r2 = h2.add_run("2. Nhóm Hệ Thống Thuật Toán & Tổng Hợp (Algorithmic Systems)")
    r2.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    r2.font.name = "Calibri"

    # Nhánh 1: Thiên về Server-side
    add_slide_card(doc, "Nhánh 1 — Khử Client Drift & Lệch Nhãn (Thiên Về Xử Lý Phía Server / Global-Centric)", [
        ("Bản chất & Kiến trúc", "Mục tiêu tối thượng là xây dựng MỘT MÔ HÌNH TOÀN CỤC DUY NHẤT (Single Global Model). Do đó, trọng tâm tối ưu nằm ở phía MÁY CHỦ (Server-side): Server tận dụng góc nhìn toàn cảnh (Global View) để điều phối, khử trôi dạt gradient (Client Drift), chuẩn hóa trọng số tổng hợp hoặc chia sẻ tri thức toàn cục."),
        ("FedAvg (McMahan et al., 2017)", "Baseline kinh điển: Server trung bình trọng số theo số lượng mẫu. Dễ mất ổn định khi Non-IID cao."),
        ("FedProx (Li et al., 2020)", "Bổ sung thành phần điều chuẩn Proximal Term: ghìm khoảng cách giữa trọng số client và mô hình Server toàn cục, chịu lỗi tốt khi có client tính toán chậm (Stragglers)."),
        ("SCAFFOLD (Karimireddy et al., 2020)", "Server duy trì và cập nhật biến đối chứng toàn cục (Global Control Variates) để ước lượng hướng trôi dạt trung bình và bù trừ gradient cho các client."),
        ("FedNova (Wang et al., 2020)", "Server chuẩn hóa và tái cân bằng trọng số tổng hợp theo số bước tính toán thực tế của từng client, tránh thiên lệch về phía client chạy nhiều bước."),
        ("Server Momentum & Adaptive FL (FedAvgM, FedAdam)", "Server coi vector tổng hợp như một gradient giả và tối ưu hóa bước cập nhật toàn cục bằng momentum/adaptive learning rate."),
        ("FedSLD (Luo & Wu, ISBI 2022 - Đề tài tham khảo)", "Server tổng hợp tỷ lệ các nhóm bệnh thành Phân Phối Nhãn Toàn Cục (Shared Label Distribution) rồi phát xuống để từng client căn chỉnh hàm mất mát (Logit Adjustment), giải quyết triệt để vấn đề lệch nhãn trong ảnh y tế.")
    ], bg_color="F4F8FA", border_color="0078D7")

    # Nhánh 2: Thiên về Client-side
    add_slide_card(doc, "Nhánh 2 — Xử Lý Lệch Đặc Trưng & Cá Nhân Hóa (Thiên Về Xử Lý Phía Client / Client-Centric)", [
        ("Bản chất & Kiến trúc", "Thừa nhận rằng một mô hình toàn cục duy nhất không thể tối ưu cho mọi bệnh viện do khác biệt thiết bị chụp (Scanner Bias) và đặc thù bệnh nhân. Trọng tâm can thiệp nằm ở phía BỆNH VIỆN (Client-side): Từng client chủ động giữ lại hoặc tinh chỉnh các thành phần mô hình cho riêng mình."),
        ("Local Fine-Tuning", "Huấn luyện mô hình toàn cục qua FL xong, mỗi client tự chạy thêm vài epoch tinh chỉnh trên dữ liệu nội bộ của riêng mình."),
        ("Model Splitting (FedRep, FedPer)", "Client giữ riêng đầu phân loại cục bộ (Personalized Head) phù hợp cơ cấu bệnh viện, chỉ gửi phần trích xuất đặc trưng chung (Shared Backbone) lên Server."),
        ("Domain Normalization (FedBN - Li et al., 2021)", "Client giữ riêng các lớp Batch Normalization cục bộ để triệt tiêu nhiễu ánh sáng/độ tương phản máy chụp, chỉ chia sẻ các tầng Convolution lên Server."),
        ("Meta-Learning (Per-FedAvg, pFedHN)", "Áp dụng tư tưởng MAML: Huấn luyện mô hình khởi tạo sao cho mỗi client chỉ cần 1 vài bước gradient cục bộ là thích ứng tối ưu nhất với dữ liệu riêng.")
    ], bg_color="FAF6F2", border_color="D97706")

    # ─────────────────────────────────────────────────────────────
    # PHẦN 3: NỀN TẢNG PHẦN MỀM & LÝ DO CHỌN FLOWER
    # ─────────────────────────────────────────────────────────────
    h3 = doc.add_heading(level=1)
    r3 = h3.add_run("3. Nền Tảng Phần Mềm & Lý Do Đồ Án Lựa Chọn Flower (flwr)")
    r3.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    r3.font.name = "Calibri"

    # Bảng 2: So sánh đa chiều nền tảng phần mềm
    t_plat = doc.add_table(rows=5, cols=4)
    t_plat.style = 'Table Grid'
    t_plat.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_plat, color="B0C4DE")

    col_widths_plat = [Inches(1.2), Inches(1.9), Inches(1.85), Inches(1.85)]

    plat_headers = [
        "Nền Tảng FL",
        "Khả Năng Tùy Biến Thuật Toán (Custom Strategy)",
        "Mức Độ Dễ Học & Tiếp Cận (Learning Curve)",
        "Mô Phỏng & Triển Khai Thực Tế (Simulation to Edge)"
    ]
    plat_rows = [
        [
            "NVIDIA FLARE\n(NVIDIA)",
            "• Phức tạp & Rườm rà: Kiến trúc Event-driven Controller/Executor đòi hỏi nhiều mã nguồn khung (boilerplate), rất khó can thiệp sâu để thử nghiệm giải thuật mới.",
            "• Rất khó tiếp cận: Đường cong học tập dốc, cấu hình hệ thống nặng nề, phụ thuộc chặt chẽ vào phần cứng GPU và hệ sinh thái của NVIDIA.",
            "• Tối ưu mạnh cho cụm GPU máy chủ lớn; triển khai xuống thiết bị nhúng nhỏ (Jetson) đòi hỏi cấu hình container cồng kềnh."
        ],
        [
            "OpenFL\n(Intel / Linux Fdn)",
            "• Cứng nhắc & Hạn chế: Quy trình tổng hợp bị đóng khung theo chuẩn bảo mật, khó chỉnh sửa logic tổng hợp lõi hoặc áp dụng các hàm loss hiệu chỉnh mới.",
            "• Khó tiếp cận: Đòi hỏi thiết lập hạ tầng chứng chỉ bảo mật (PKI/TLS) phức tạp ngay từ đầu; tài liệu ít và cộng đồng hỗ trợ hạn chế.",
            "• Tập trung tốt cho kết nối liên viện Cross-Silo; khả năng mô phỏng linh hoạt nhiều client trên 1 máy tính đơn lẻ còn yếu."
        ],
        [
            "Fed-BioMed\n(INRIA Pháp)",
            "• Rất khó mở rộng: Chủ yếu đóng gói sẵn một số thuật toán cơ bản, không được thiết kế tối ưu cho việc tùy biến và nghiên cứu thuật toán mới.",
            "• Khó tiếp cận: Bắt buộc cài đặt Docker và cấu hình node bệnh viện ảo, nặng về kỹ năng quản trị hạ tầng IT hơn là nghiên cứu ML.",
            "• Chuẩn mực tốt cho thử nghiệm lâm sàng thực tế tại bệnh viện, nhưng tốc độ mô phỏng thử nghiệm học thuật rất chậm."
        ],
        [
            "Flower (flwr)\n★ Lựa chọn của đồ án",
            "• Tối ưu & Linh hoạt nhất: Kế thừa trực tiếp Strategy, override hàm aggregate_fit là hiện thực hóa FedProx, FedNova, FedSLD chỉ với vài chục dòng code.",
            "• Thân thiện nhất cho người mới: Cực kỳ lý tưởng cho người mới nghiên cứu, học tập FL; API Pythonic trực quan, tài liệu phong phú, chạy ngay trên PC/Laptop.",
            "• Chuyển tiếp liền mạch (100% code reuse): Chạy mô phỏng 10-50 viện trên 1 GPU (Virtual Client Engine) rồi mang nguyên code sang cụm Jetson / Bệnh viện thực tế."
        ]
    ]

    for c_i, h_text in enumerate(plat_headers):
        c = t_plat.rows[0].cells[c_i]
        c.width = col_widths_plat[c_i]
        c.text = h_text
        set_cell_background(c, "1F497D")
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = "Calibri"
            r.font.size = Pt(9.5)
            r.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    for r_i, r_data in enumerate(plat_rows):
        row = t_plat.rows[r_i + 1]
        for c_i, val in enumerate(r_data):
            cell = row.cells[c_i]
            cell.width = col_widths_plat[c_i]
            cell.text = val
            set_cell_margins(cell, top=80, bottom=80, left=110, right=110)
            if r_i == 3: # Flower
                set_cell_background(cell, "EBF3FB")
            elif r_i % 2 == 1:
                set_cell_background(cell, "F7F9FB")
            p = cell.paragraphs[0]
            for r in p.runs:
                r.font.name = "Calibri"
                r.font.size = Pt(8.5)
                if c_i == 0:
                    r.bold = True
                    r.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    from pathlib import Path
    output_paths = [Path("Document/review_2.docx"), Path("review_2.docx")]
    for out in output_paths:
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            doc.save(str(out))
            print("Created successfully: " + str(out))
        except Exception as e:
            print(f"Notice for {out}: {e}")

if __name__ == "__main__":
    build_review_2_docx()
