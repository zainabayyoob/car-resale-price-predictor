# Automotive car Price Predictor using ML🚗📊

An enterprise-grade, end-to-end Machine Learning pipeline designed to ingest raw, unformatted car marketplace listings, execute an automated data cleaning sequence, and train a robust **Linear Regression** model using Scikit-Learn pipelines. The system features an interactive command-line interface (CLI) for real-time vehicle valuation.

---

## 🚀 Key Architectural Features
* **Automated Data Cleaning:** Programmatically handles null values, strips currency/text anomalies from price and distance attributes, filters statistical outliers, and standardizes vehicle string formats.
* **Encapsulated ML Pipeline:** Leverages Scikit-Learn's `make_column_transformer` and `OneHotEncoder` within a unified pipeline wrapper to prevent data leakage and handle categorical features gracefully.
* **Exploratory Data Analysis (EDA):** Automatically generates and saves distribution analytics (`car_project_outputs/01_price_distribution.png`) upon execution.

---

## 📊 Exploratory Data Analysis
The framework automatically analyzes the processed data and exports visual distribution matrices to evaluate pricing variance across leading manufacturers:

![Price Distribution Across Top Brands](car_project_outputs/01_price_distribution.png)

---

## 💡 SYSTEM INPUT RULES & INFERENCE SPECIFICATIONS

Because the underlying model relies on structural categorical encoding (`OneHotEncoder`), the real-time inference engine enforces strict syntax validation. **Please review these rules before submitting inputs to avoid prediction errors.**

### 🔴 Core Formatting Requirements
1. **Strict Case Sensitivity:** Categorical inputs must follow standard Title Case (e.g., input **`Toyota`**, do **NOT** input `toyota` or `TOYOTA`).
2. **The First-Word Prefix Rule:** The **Car Model Name (Input 1)** *must* always begin with the exact name of the **Company Brand (Input 2)**. 
   * *Correct Alignment:* Model: `Hyundai i20 Asta` | Brand: `Hyundai`
   * *Incorrect Alignment:* Model: `i20 Asta` | Brand: `Hyundai` (This breaks the encoding arrays)
3. **Clean Numerical Ingestion:** Do not include structural delimiters, symbols, commas, or unit text when entering data for Manufacturing Year or Kilometers Driven.
   * *Correct:* `45000` | `2017`
   * *Incorrect:* `45,000 kms` | `Year 2017`

---

### 📋 Verified Integration Examples (Ready for Quick Copy-Paste)

To evaluate the predictive performance of the model immediately, copy and paste these pre-validated data profiles directly into your terminal prompts:

#### 🔹 Profile A: Premium Hatchback
* **1. Car Model Name:** `Hyundai i20 Asta`
* **2. Company Brand:** `Hyundai`
* **3. Manufacturing Year:** `2016`
* **4. Kilometers Driven:** `50000`
* **5. Fuel Type:** `Petrol`

#### 🔹 Profile B: Sports Utility Vehicle (SUV)
* **1. Car Model Name:** `Mahindra XUV500 W8`
* **2. Company Brand:** `Mahindra`
* **3. Manufacturing Year:** `2015`
* **4. Kilometers Driven:** `70000`
* **5. Fuel Type:** `Diesel`

#### 🔹 Profile C: Mass-Market Budget Sedan
* **1. Car Model Name:** `Maruti Suzuki Swift`
* **2. Company Brand:** `Maruti`
* **3. Manufacturing Year:** `2017`
* **4. Kilometers Driven:** `35000`
* **5. Fuel Type:** `Petrol`

---

## 💻 Local Deployment & Execution

### Prerequisites
Ensure you have Python 3.8+ installed along with the required analytical libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
