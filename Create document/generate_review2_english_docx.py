"""
generate_review2_english_docx.py
Script to generate review_2_english.docx:
Accurate, slide-ready, high-density English summary covering:
1. Multi-dimensional taxonomy of FL systems (Topology, Scale, Non-IID taxonomy from Kairouz et al. 2021)
2. Algorithmic systems (Client Drift & Label Skew vs. Feature Shift & Personalized FL)
3. Software platform comparison & 4 core rationales for choosing Flower (flwr)
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
    """Adds a slide-card box with bullet points."""
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

def build_review_2_english_docx():
    doc = docx.Document()
    
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)
        
    # Title
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(3)
    title_p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = title_p.add_run("REVIEW 2: FEDERATED LEARNING TAXONOMY & PLATFORM SELECTION")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(15.5)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    
    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(12)
    sub_p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = sub_p.add_run("Slide-Ready Academic Synthesis Document\nDirectly referenced from: \"Advances and Open Problems in Federated Learning\" (Kairouz et al., 2021)")
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(11)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(0x59, 0x59, 0x59)

    # ─────────────────────────────────────────────────────────────
    # PART 1: MULTI-DIMENSIONAL TAXONOMY OF FL SYSTEMS
    # ─────────────────────────────────────────────────────────────
    h1 = doc.add_heading(level=1)
    r1 = h1.add_run("1. Multi-Dimensional Taxonomy of Existing FL Systems")
    r1.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    r1.font.name = "Calibri"

    # 1.1 Architecture & Topology
    add_slide_card(doc, "1.1. Architecture & Network Topology: Centralized FL vs. Decentralized FL", [
        ("Centralized FL (Server–Client / Hub-and-Spoke)", "Coordinated by a central server (Parameter Server): handles global model distribution, client selection, and weight aggregation (e.g., FedAvg). Pros: Fast empirical convergence, simple governance, easy security verification. Cons: Single Point of Failure (SPoF) and server bandwidth bottleneck."),
        ("Decentralized FL (Peer-to-Peer / Gossip-based)", "No central coordinator. Clients communicate model updates directly with topological neighbors via graph consensus protocols. Pros: Eliminates SPoF, maximal organizational autonomy. Cons: Slower convergence rate, high peer-to-peer communication overhead, complex verification."),
        ("Medical Context Choice", "Cross-hospital healthcare federations predominantly employ Centralized FL with Secure Aggregation protocols to satisfy institutional compliance and clinical governance.")
    ])

    # 1.2 Deployment Scale
    add_slide_card(doc, "1.2. Deployment Scale & Infrastructure: Cross-Silo FL vs. Cross-Device FL (Table 1 - Kairouz et al.)", [
        ("Cross-Silo FL (Inter-Institutional / Hospital Setting)", "Scale: 2 – 100 organizations/hospitals/datacenters. Hardware: High-performance workstations, clinical-grade GPUs, large memory. Network: High-bandwidth, reliable WAN. Availability: Almost 100% online across all rounds (Stateful clients). Partitioning: Horizontal (sample-partitioned) or Vertical (feature-partitioned)."),
        ("Cross-Device / Edge FL (Massive Mobile / IoT Setting)", "Scale: 10^3 – 10^10 consumer mobile or IoT edge devices. Hardware: Resource-constrained (battery, compute, RAM). Network: Intermittent Wi-Fi/cellular. Availability: <1% active per round with high drop-out rates (Stateless clients)."),
        ("FedMedAI Project Positioning", "Our thesis is primarily designed for Cross-Silo Medical FL (inter-hospital collaborative learning) with native extensibility toward Edge Medical FL (embedded edge workstations / NVIDIA Jetson).")
    ])

    # Table 1: Cross-Silo vs Cross-Device
    t_scale = doc.add_table(rows=6, cols=3)
    t_scale.style = 'Table Grid'
    t_scale.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_scale, color="B0C4DE")

    scale_headers = ["Evaluation Metric", "Cross-Silo FL (FedMedAI Setting)", "Cross-Device / Edge FL"]
    scale_rows = [
        ["Participating Clients", "2 – 100 Hospitals / Research Institutes", "10^3 – 10^10 Mobile / IoT edge devices"],
        ["Compute Resources", "High-performance servers, medical GPUs", "Low-power mobile chips (battery/RAM constraints)"],
        ["Client Availability & State", "Nearly 100% online, stateful across rounds", "<1% selected per round, high dropout (stateless)"],
        ["Primary Bottleneck", "Computation & institutional compliance", "Communication bandwidth (network latency)"],
        ["Application Objective", "High-precision clinical diagnostic AI", "Next-word prediction, consumer recommendations"]
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

    # 1.3 Non-IID Taxonomy (Plain clinical language, no complex math formulas)
    add_slide_card(doc, "1.3. Data Partitioning: Non-IID Regimes in Healthcare", [
        ("1. Disease Prevalence Imbalance (Label Distribution Skew)", 
         "The proportion of disease categories varies substantially across clinical sites. Clinical example: An Oncology Hospital sees predominantly malignant tumor cases, whereas a General Community Clinic sees mostly benign, routine conditions and very few tumors."),
        ("2. Imaging Variation Across Scanners (Feature Distribution Skew / Scanner Bias)", 
         "The exact same disease pathology produces divergent image intensity, contrast, and noise levels due to hardware differences across scanner vendors (Siemens, GE, Philips)."),
        ("3. Diagnostic and Manifestation Variations (Concept Drift)", 
         "• Same disease, differing appearance: Identical pneumonia shows markedly different radiographic patterns in pediatric vs. geriatric cohorts.\n"
         "• Same appearance, differing diagnosis: Identical tissue morphology evaluated under differing diagnostic threshold guidelines across regional healthcare systems."),
        ("4. Patient Volume Imbalance (Quantity Skew)", 
         "Clinical data volume differs by orders of magnitude: Major central university hospitals hold hundreds of thousands of cases, while local community clinics hold only dozens."),
        ("5. Data Partitioning in the FedMedAI Project",
         "• Standardized Benchmarks: Primary validation on OrganMNIST-Axial (11 abdominal organ CT classes, 58K images); extensible to and benchmarked against BloodMNIST (8 blood cell types) and DermaMNIST (7 skin lesion classes).\n"
         "• Targeted Non-IID Regime: Directly engineered to resolve Disease Prevalence Imbalance (Label Distribution Skew) — the primary multi-center clinical barrier addressed by our investigated strategy (FedSLD via shared label distributions and classification calibration).\n"
         "• Empirical Scenarios (12 Hospitals / Clients):\n"
         "   - Practical Scenario (Practical Non-IID): Simulates multi-specialty general hospitals treating multiple diseases with highly unbalanced disease prevalence.\n"
         "   - Extreme Sub-specialty Scenario (Pathological Non-IID): Each clinic only manages 1 or 2 restricted disease classes, pushing algorithmic stability against Client Drift to its limits.")
    ])

    # ─────────────────────────────────────────────────────────────
    # PART 2: ALGORITHMIC SYSTEMS & AGGREGATION
    # ─────────────────────────────────────────────────────────────
    h2 = doc.add_heading(level=1)
    r2 = h2.add_run("2. Algorithmic Systems & Optimization Paradigms")
    r2.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    r2.font.name = "Calibri"

    # Branch 1: Server-centric
    add_slide_card(doc, "Branch 1 — Mitigating Client Drift & Label Skew (Server-Centric / Global Optimization)", [
        ("Architecture & Philosophy", "The fundamental objective is to train a SINGLE GLOBAL CONSENSUS MODEL. Consequently, the primary optimization locus resides on the SERVER (Global-side): The server leverages its global vantage point to orchestrate parameter updates, eliminate Client Drift, normalize aggregated weights, or broadcast global distribution priors."),
        ("FedAvg (McMahan et al., 2017)", "Classical baseline: Server executes sample-weighted parameter averaging. Prone to severe drift under high Non-IID skew."),
        ("FedProx (Li et al., 2020)", "Introduces a proximal regularization term: restricts local model updates from straying too far from the global server parameters; tolerates system stragglers."),
        ("SCAFFOLD (Karimireddy et al., 2020)", "Server maintains and updates Global Control Variates to explicitly estimate drift directions and correct client gradient paths."),
        ("FedNova (Wang et al., 2020)", "Server normalizes and rescales aggregation weights based on each client's actual computational steps, avoiding bias toward clients executing more local epochs."),
        ("Server Momentum & Adaptive FL (FedAvgM, FedAdam)", "Server treats aggregated client updates as a pseudo-gradient, optimizing the global update trajectory with momentum and adaptive learning rates."),
        ("FedSLD (Luo & Wu, ISBI 2022 - Project Reference)", "Server aggregates client class frequencies into a Shared Label Distribution and broadcasts it back to calibrate local loss functions (Logit Adjustment), thoroughly resolving multi-center label skew.")
    ], bg_color="F4F8FA", border_color="0078D7")

    # Branch 2: Client-centric
    add_slide_card(doc, "Branch 2 — Handling Scanner Bias & Personalized FL (Client-Centric / Local Adaptation)", [
        ("Architecture & Philosophy", "Acknowledges that a single global model cannot be universally optimal across disparate clinical sites due to hardware differences (Scanner Bias) and distinct patient cohorts. The optimization locus is situated at the HOSPITAL (Client-side): Individual clients retain or tailor model components locally rather than forcing total global consensus."),
        ("Local Fine-Tuning", "Following global FL convergence, each client executes localized training epochs on its proprietary internal data."),
        ("Model Splitting (FedRep, FedPer)", "Clients retain private Personalized Classification Heads suited to their local demographics, sending only the Shared Feature Backbone to the server."),
        ("Domain Normalization (FedBN - Li et al., 2021)", "Clients maintain local Batch Normalization layers to neutralize scanner contrast and intensity shifts, sharing only Convolutional parameters with the server."),
        ("Meta-Learning (Per-FedAvg, pFedHN)", "Adopts MAML principles: Learns an optimal global initialization from which individual clients can achieve peak adaptation within 1–3 local gradient steps.")
    ], bg_color="FAF6F2", border_color="D97706")

    # ─────────────────────────────────────────────────────────────
    # PART 3: SOFTWARE PLATFORMS & RATIONALE FOR FLOWER
    # ─────────────────────────────────────────────────────────────
    h3 = doc.add_heading(level=1)
    r3 = h3.add_run("3. Software Framework Comparison & Rationale for Flower (flwr)")
    r3.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    r3.font.name = "Calibri"

    # Table 2: Multi-dimensional Platform Comparison
    t_plat = doc.add_table(rows=5, cols=4)
    t_plat.style = 'Table Grid'
    t_plat.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(t_plat, color="B0C4DE")

    col_widths_plat = [Inches(1.2), Inches(1.9), Inches(1.85), Inches(1.85)]

    plat_headers = [
        "FL Framework",
        "Custom Strategy Extensibility (Novel Algorithms)",
        "Accessibility for Learners & Newcomers (Learning Curve)",
        "Simulation to Edge Deployment (Workflow)"
    ]
    plat_rows = [
        [
            "NVIDIA FLARE\n(NVIDIA)",
            "• Complex & Boilerplate-heavy: Event-driven Controller/Executor architecture introduces high code overhead, making it difficult to inject custom novel loss/aggregation logic.",
            "• Steep Learning Curve: Heavy enterprise infrastructure footprint, complex daemon setup, and strict dependency on the proprietary NVIDIA GPU hardware stack.",
            "• Optimized for large-scale GPU server clusters; deploying onto lightweight edge endpoints (e.g., Jetson) requires heavy containerized configurations."
        ],
        [
            "OpenFL\n(Intel / Linux Fdn)",
            "• Rigid & Constrained: Aggregation pipeline is strictly locked into predefined security workflows, making modifications to core aggregation logic cumbersome.",
            "• High Setup Barrier: Mandates complex manual setup of mutual TLS/PKI certificates from day one; documentation is sparse and the developer community is small.",
            "• Well-tailored for hospital Cross-Silo deployments; single-machine local multi-client simulation is rigid and resource-inefficient."
        ],
        [
            "Fed-BioMed\n(INRIA France)",
            "• Hard to Extend: Primarily packaged with a few pre-built standard algorithms; not architected for prototyping novel optimization or calibration strategies.",
            "• High Entry Barrier: Requires mandatory Docker daemon setups and virtual hospital node configurations, focusing more on IT/DevOps than ML research.",
            "• Purpose-built for actual clinical trials in hospital networks, but iterative algorithmic simulation speed is sluggish for academic prototyping."
        ],
        [
            "Flower (flwr)\n★ Thesis Selection",
            "• Unrivaled & Easiest Extensibility: Subclasses Strategy directly; overriding aggregate_fit implements FedProx, FedNova, and FedSLD in minimal lines of code.",
            "• Most Accessible for Learners: Gentle learning curve, clean Pythonic API, extensive community tutorials; installs and runs instantly on PC/laptops without hurdles.",
            "• Seamless Transition (100% Code Reuse): Simulates 10–50 hospital clients on 1 GPU (Virtual Client Engine), then deploys the exact same codebase to physical Jetson / hospital nodes."
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

    from pathlib import Path
    output_paths = [Path("Document/review_2_english.docx"), Path("review_2_english.docx")]
    for out in output_paths:
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            doc.save(str(out))
            print("Created successfully: " + str(out))
        except Exception as e:
            print(f"Notice for {out}: {e}")

if __name__ == "__main__":
    build_review_2_english_docx()
