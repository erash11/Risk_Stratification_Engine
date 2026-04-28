# Local AI Workstation & Sports Science Analytics Blueprint

## 1. Executive Summary
This document outlines the strategic hardware and software requirements for deploying a completely private, high-capacity local AI workstation. It additionally details the roadmap for utilizing this local AI to architect a predictive injury risk stratification engine using high-performance athlete data.

---

## 2. Hardware Architecture: The Multi-GPU Workstation
To achieve frontier-level intelligence (running 70B to 104B parameter models like Llama 3.1 70B or Qwen 2.5 72B locally at 4-bit/5-bit quantization), a custom multi-GPU PC build is the most cost-effective and performant solution.

### Recommended Specifications & Estimated Pricing (2026 Target)
* **GPUs:** 2x Used Nvidia RTX 3090 24GB (~$1,900 - $2,200)
* **CPU:** AMD Ryzen 9 7900X or 7950X (~$400 - $500)
* **Motherboard:** ASUS ProArt X670E-Creator WiFi (optimal PCIe spacing) (~$450)
* **RAM:** 128GB DDR5 5200MHz+ (~$350)
* **Storage:** 4TB NVMe M.2 Gen4 SSD (~$250)
* **Power Supply:** 1600W 80+ Titanium ATX PSU (e.g., Corsair AX1600i) (~$450)
* **Case & Cooling:** High airflow case (Corsair 7000D / Fractal Torrent) + 360mm AIO Liquid Cooler (~$400)
* **Total Estimated Cost:** **$4,200 - $4,650**

---

## 3. Software Stack & Deployment Steps
The software environment isolates the backend generation engine from the frontend chat interface, prioritizing speed and flexibility.

1. **Operating System:** Ubuntu Linux 24.04 LTS (highly recommended for seamless multi-GPU memory management).
2. **Drivers:** Nvidia Proprietary Drivers and CUDA Toolkit.
3. **Containerization:** Docker (to isolate and manage local AI services).
4. **Backend Inference Engine:** **Ollama** (handles model weights, GPU splitting, and VRAM allocation).
5. **Frontend Interface:** **Open WebUI** (provides a polished, modern chat interface comparable to commercial AI products, with built-in RAG capabilities).
6. **Target Models (Quantized GGUF/EXL2):** Llama-3.1-70B-Instruct, Qwen-2.5-72B, or Command R+ (104B).

---

## 4. Analytical Engine Blueprint: Injury Risk Stratification
The ultimate application of this local AI system is to serve as a coding co-pilot to build a machine learning model capable of mining athlete datasets and identifying elevated injury risk profiles.

### Phase 1: Centralizing the Data Lake
Unify distinct data streams into a master table (Pandas DataFrame) indexed by `Athlete_ID` and `Date`:
* **External Load:** GPS yardage, high-speed distance, accelerations/decelerations.
* **Internal Load:** Cardiovascular exertion via optical HR monitors.
* **Neuromuscular / Kinetics:** Countermovement jumps (force plates), eccentric hamstring metrics (NordBord/groin testing), and kinematic markers (motion capture).
* **Structural Baseline:** Tissue and body composition (DEXA scans).

### Phase 2: Feature Engineering (The Predictive Signals)
Utilize the AI to write Python scripts that transform raw daily numbers into rolling trend features:
* **Acute:Chronic Workload Ratio (ACWR):** Exponential weighted moving averages of high-speed running (7-day vs. 28-day).
* **Kinetic Variance:** Coefficient of Variation (COV) in peak landing forces across a rolling 14-day window.
* **Load Discrepancy:** Divergence between external load stability and internal load spikes.

### Phase 3: Model Architecture & Explainability
* **The Models:** Supervised tree-based algorithms like **Random Forest** or **XGBoost**. These models handle tabular data and missing variables efficiently. Deep learning/neural networks should be avoided due to their "black-box" nature.
* **The Explainability Layer:** Implement **SHAP (SHapley Additive exPlanations)** values to break down exactly *why* a specific athlete's risk score is elevated (e.g., flagging a concurrent drop in concentric power and spike in player load), allowing for actionable interventions by performance and sport coaches.

---

## 5. Community & Educational Resources
To accelerate the setup and optimization phases, the following industry leaders and creators are highly recommended:
* **YouTube:** Matthew Berman (Local LLM tutorials), Techno Tim (Hardware & server infrastructure), Cole Medin (Technical LLM architecture).
* **LinkedIn:** Matt Shumer (Model optimization and local inference hardware), Maxime Labonne (Quantization and model merging).
