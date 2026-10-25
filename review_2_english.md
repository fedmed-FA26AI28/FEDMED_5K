# REVIEW 2: FEDERATED LEARNING TAXONOMY & PLATFORM SELECTION
*(Slide-Ready Synthesis for Thesis Presentation & Committee Defense)*

> **Core Theoretical Reference:**  
> Peter Kairouz, H. Brendan McMahan et al., *"Advances and Open Problems in Federated Learning"*, Foundations and Trends in Machine Learning, 2021 (arXiv:1912.04977).

---

## SLIDE 1: Multi-Dimensional Taxonomy of Federated Learning Systems

### 1.1. Architecture & Topology (Network Architecture & Topology)
* **Centralized FL (Server–Client / Hub-and-Spoke)**:
  - **Mechanism**: Orchestrated by a central Parameter Server: manages global model distribution, client selection, and parameter aggregation (e.g., FedAvg).
  - **Advantages**: Fast empirical convergence, simple governance, easy security verification and parameter synchronization.
  - **Limitations**: Single Point of Failure (SPoF) and server bandwidth bottleneck.
  - *Clinical Application*: Predominant standard for hospital federations under regulatory governance.
* **Decentralized FL (Peer-to-Peer / Gossip-based)**:
  - **Mechanism**: No central coordinator. Clients communicate directly with topological neighbors via graph consensus protocols.
  - **Advantages**: Completely eliminates SPoF, maximizes organizational autonomy among participants.
  - **Limitations**: Slower convergence rate, high peer-to-peer communication overhead, complex verification and quality control.

---

### 1.2. Deployment Scale: Cross-Silo FL vs. Cross-Device FL *(Table 1 - Kairouz et al.)*

| Comparison Criteria | Cross-Silo FL *(FedMedAI Setting)* | Cross-Device / Edge FL |
| :--- | :--- | :--- |
| **Participating Clients** | **2 – 100 Hospitals / Research Centers** | Millions of mobile / IoT devices |
| **Compute Resources** | **High**: Dedicated servers, clinical-grade GPUs | **Low**: Resource-constrained mobile chips (battery/RAM) |
| **Availability & State** | **Nearly 100% online across rounds** *(Stateful)* | <1% active per round, high dropout rates *(Stateless)* |
| **Primary Bottleneck** | Computation & regulatory verification | Communication bandwidth *(network latency)* |
| **Target Application** | **High-precision clinical diagnostic AI** | Next-word prediction, consumer recommendations |

> **Slide Takeaway:** The **FedMedAI** project is strictly positioned within **Cross-Silo Medical FL** (collaborative inter-hospital training) with architectural support extending to **Edge Medical FL** (embedded clinical edge devices such as NVIDIA Jetson).

---

### 1.3. Data Partitioning: Non-IID Regimes in Healthcare

* **1. Disease Prevalence Imbalance (Label Distribution Skew)**:
  - The proportion of disease categories varies substantially across clinical sites.
  - *Clinical Example*: A Specialized Oncology Hospital sees predominantly malignant tumor cases, whereas a General Community Clinic sees mostly benign, routine conditions and very few tumors.
* **2. Imaging Variation Across Scanners (Feature Distribution Skew / Scanner Bias)**:
  - The exact same disease pathology produces divergent image intensity, contrast, and noise levels due to hardware differences across scanner vendors (Siemens, GE, Philips).
* **3. Diagnostic and Manifestation Variations (Concept Drift)**:
  - *Same disease, differing appearance*: Identical pneumonia shows markedly different radiographic patterns in pediatric vs. geriatric cohorts.
  - *Same appearance, differing diagnosis*: Identical tissue morphology evaluated under differing diagnostic threshold guidelines across regional healthcare systems.
* **4. Patient Volume Imbalance (Quantity Skew)**:
  - Clinical data volume differs by orders of magnitude: Major central university hospitals hold hundreds of thousands of cases, while local community clinics hold only dozens.
* **5. Data Partitioning in the FedMedAI Project**:
  - **Standardized Benchmarks**: Primary validation on **OrganMNIST-Axial** (11 abdominal organ CT classes, 58K images); extensible to and benchmarked against **BloodMNIST** (8 blood cell types) and **DermaMNIST** (7 skin lesion classes).
  - **Core Non-IID Target**: Directly addresses **Disease Prevalence Imbalance (Label Distribution Skew)** — the key multi-center obstacle resolved by our proposed **FedSLD** strategy via shared label distributions and classification calibration.
  - **Empirical Scenarios (12 Hospitals / Clients)**:
    1. *Practical Scenario (Practical Non-IID)*: Simulates multi-specialty general hospitals treating multiple diseases with highly unbalanced disease prevalence.
    2. *Extreme Sub-specialty Scenario (Pathological Non-IID)*: Each clinic only manages 1 or 2 restricted disease classes, pushing algorithmic stability against Client Drift to its limits.

---

## SLIDE 2: Algorithmic Systems & Optimization Paradigms

### Branch 1 — Mitigating Client Drift & Label Skew (Server-Centric / Global Optimization)
* **Architecture & Philosophy**: The fundamental objective is to train a **SINGLE GLOBAL CONSENSUS MODEL**. Consequently, the primary optimization locus resides on the **SERVER (Global-side)**: The server leverages its global vantage point to orchestrate parameter updates, eliminate Client Drift, normalize aggregated weights, or broadcast global distribution priors.
* **Key Algorithmic Paradigms**:
  - **FedAvg** *(McMahan et al., 2017)*: Classical baseline, Server executes sample-weighted parameter averaging. Prone to severe drift under high Non-IID skew.
  - **FedProx** *(Li et al., 2020)*: Introduces a proximal regularization term restricting local updates from straying too far from the global server parameters; tolerates system stragglers.
  - **SCAFFOLD** *(Karimireddy et al., 2020)*: Server maintains and updates Global Control Variates to explicitly estimate drift directions and correct client gradient paths.
  - **FedNova** *(Wang et al., 2020)*: Server normalizes and rescales aggregation weights based on each client's actual computational steps, avoiding bias toward clients executing more local epochs.
  - **Server Momentum & Adaptive FL** *(FedAvgM, FedAdam)*: Server treats aggregated client updates as a pseudo-gradient, optimizing the global update trajectory with momentum and adaptive learning rates.
  - **FedSLD** *(Luo & Wu, ISBI 2022 - Project Reference)*: Server aggregates client class frequencies into a **Shared Label Distribution** and broadcasts it back to calibrate local loss functions (*Logit Adjustment*), thoroughly resolving multi-center label skew.

---

### Branch 2 — Handling Scanner Bias & Personalized FL (Client-Centric / Local Adaptation)
* **Architecture & Philosophy**: Acknowledges that a single global model cannot be universally optimal across disparate clinical sites due to hardware differences (**Scanner Bias**) and distinct patient cohorts. The optimization locus is situated at the **HOSPITAL (Client-side)**: Individual clients retain or tailor model components locally rather than forcing total global consensus.
* **Key Algorithmic Paradigms**:
  - **Local Fine-Tuning**: Following global FL convergence, each client executes localized training epochs on its proprietary internal data.
  - **Model Splitting (FedRep, FedPer)**: Clients retain private Personalized Classification Heads suited to their local demographics, sending only the Shared Feature Backbone to the server.
  - **Domain Normalization (FedBN - Li et al., 2021)**: Clients maintain local Batch Normalization layers to neutralize scanner contrast and intensity shifts, sharing only Convolutional parameters with the server.
  - **Meta-Learning (Per-FedAvg, pFedHN)**: Adopts MAML principles: Learns an optimal global initialization from which individual clients can achieve peak adaptation within 1–3 local gradient steps.

---

## SLIDE 3: Multi-Dimensional Software Framework Comparison & Rationale for Flower (flwr)

| FL Framework | Custom Strategy Extensibility (Novel Algorithms) | Accessibility for Learners & Newcomers (Learning Curve) | Simulation to Edge Deployment (Workflow) |
| :--- | :--- | :--- | :--- |
| **NVIDIA FLARE**<br>*(NVIDIA)* | • **Complex & Boilerplate-heavy**: Event-driven Controller/Executor architecture introduces high code overhead, making it difficult to inject custom novel loss/aggregation logic. | • **Steep Learning Curve**: Heavy enterprise infrastructure footprint, complex daemon setup, and strict dependency on the proprietary NVIDIA GPU hardware stack. | • Optimized for large-scale GPU server clusters; deploying onto lightweight edge endpoints (e.g., Jetson) requires heavy containerized configurations. |
| **OpenFL**<br>*(Intel / Linux Fdn)* | • **Rigid & Constrained**: Aggregation pipeline is strictly locked into predefined security workflows, making modifications to core aggregation logic cumbersome. | • **High Setup Barrier**: Mandates complex manual setup of mutual TLS/PKI certificates from day one; documentation is sparse and the developer community is small. | • Well-tailored for hospital Cross-Silo deployments; single-machine local multi-client simulation is rigid and resource-inefficient. |
| **Fed-BioMed**<br>*(INRIA France)* | • **Hard to Extend**: Primarily packaged with a few pre-built standard algorithms; not architected for prototyping novel optimization or calibration strategies. | • **High Entry Barrier**: Requires mandatory Docker daemon setups and virtual hospital node configurations, focusing more on IT/DevOps than ML research. | • Purpose-built for actual clinical trials in hospital networks, but iterative algorithmic simulation speed is sluggish for academic prototyping. |
| **Flower (flwr)**<br>★ *(Thesis Selection)* | • **Unrivaled & Easiest Extensibility**: Subclasses `Strategy` directly; overriding `aggregate_fit` implements FedProx, FedNova, and FedSLD in minimal lines of code. | • **Most Accessible for Learners**: Gentle learning curve, clean Pythonic API, extensive community tutorials; installs and runs instantly on PC/laptops without hurdles. | • **Seamless Transition (100% Code Reuse)**: Simulates 10–50 hospital clients on 1 GPU (Virtual Client Engine), then deploys the exact same codebase to physical Jetson / hospital nodes. |
