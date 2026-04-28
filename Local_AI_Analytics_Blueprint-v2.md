# Local AI Sports Science Analytics: Complete Blueprint & Code Generation Prompts

## 1. Hardware Architecture: The Multi-GPU Workstation
To achieve frontier-level intelligence (running 70B to 104B parameter models like Llama 3.1 70B or Qwen 2.5 72B locally at 4-bit/5-bit quantization), a custom multi-GPU PC build is the most cost-effective solution.

### Recommended Specifications & Pricing
* **GPUs:** 2x Used Nvidia RTX 3090 24GB (~$1,900 - $2,200)
* **CPU:** AMD Ryzen 9 7900X or 7950X (~$400 - $500)
* **Motherboard:** ASUS ProArt X670E-Creator WiFi (optimal PCIe spacing) (~$450)
* **RAM:** 128GB DDR5 5200MHz+ (~$350)
* **Storage:** 4TB NVMe M.2 Gen4 SSD (~$250)
* **Power Supply:** 1600W 80+ Titanium ATX PSU (e.g., Corsair AX1600i) (~$450)
* **Case & Cooling:** High airflow case (Corsair 7000D / Fractal Torrent) + 360mm AIO Liquid Cooler (~$400)
* **Total Estimated Cost:** **$4,200 - $4,650**

## 2. Software Stack
* **Operating System:** Ubuntu Linux 24.04 LTS
* **Drivers:** Nvidia Proprietary Drivers and CUDA Toolkit
* **Containerization:** Docker
* **Backend Engine:** Ollama
* **Frontend UI:** Open WebUI
* **Target Models:** Llama-3.1-70B-Instruct, Qwen-2.5-72B, or Command R+ (104B)

---

## 3. The Analytical Engine: LLM Prompts for Python Generation

Once the local AI is running, use these exact prompts to guide the model in writing the Python scripts for your predictive engine. These prompts take your specific data streams and structure them for machine learning.

### Prompt 1: Building the Data Lake (Merging the CSVs)
**Copy/Paste this into your AI:**
> "I am a novice Python coder building a sports science predictive engine. I have historical athlete data spread across several CSV exports. I need a Python script using Pandas to merge these into a single master DataFrame. 
> 
> The datasets include:
> 1. `gps_data.csv` (Contains external load, yardage, high-speed distance)
> 2. `hr_data.csv` (Contains internal load and cardiovascular exertion from Polar optical monitors)
> 3. `force_plate.csv` (Contains countermovement jump kinetics)
> 4. `nordbord_data.csv` (Contains eccentric hamstring strength metrics)
> 5. `mocap.csv` (Contains kinematic data from markerless motion capture)
> 6. `dexa.csv` (Contains baseline body composition and tissue metrics)
> 7. `injury_log.csv` (Contains dates and binary flags for when an injury occurred: 1 for injury, 0 for healthy)
>
> All CSVs share an 'Athlete_ID' and 'Date' column, except `dexa.csv`, which is only recorded periodically. 
> 
> Please write a robust script to merge these tables on 'Athlete_ID' and 'Date'. Use a forward-fill (ffill) method for the DEXA data so the most recent scan carries forward to daily rows. Handle missing daily values (NaNs) appropriately without dropping entire rows. Keep the code heavily commented so I can understand what each block does."

### Prompt 2: Feature Engineering (Creating Predictive Signals)
**Copy/Paste this into your AI:**
> "Now that my master DataFrame is merged, I need to engineer predictive features. Raw daily data isn't enough; I need rolling trends. Please write the Pandas code to add the following columns to my dataframe:
> 
> 1. **Acute:Chronic Workload Ratio (ACWR):** Calculate the Exponential Weighted Moving Average (EWMA) of 'high_speed_distance' using a 7-day acute window and a 28-day chronic window. Return the ratio.
> 2. **Kinetic Variance:** Calculate the rolling 14-day Coefficient of Variation (COV) for 'peak_landing_force' from the force plate data.
> 3. **Internal/External Load Discrepancy:** Create a custom metric that flags when the rolling 7-day average of internal load (HR exertion) increases by more than 10%, while the external load (GPS yardage) remains flat or decreases. 
> 
> Ensure the script calculates these individually per 'Athlete_ID'."

### Prompt 3: Training the Model & Explainability
**Copy/Paste this into your AI:**
> "I now have a clean, feature-engineered dataset. I want to train an XGBoost classifier to predict the 'injury_flag' column (1 = injury, 0 = healthy). 
> 
> Please write the Python code to:
> 1. Perform a time-series train/test split (do not use random shuffling, as this is sequential time-series data).
> 2. Train an XGBoost model optimized for imbalanced data (since injuries are rare compared to healthy days).
> 3. Implement the SHAP (SHapley Additive exPlanations) library to interpret the model. 
> 4. Generate a SHAP summary plot so I can visually see which features (e.g., a drop in NordBord strength combined with an ACWR spike) are driving the most risk.
> 
> Provide the code step-by-step with explanations for a beginner."
