# 🏥 Healthcare ML System: Hospital Readmission Prediction
**Comprehensive Project Progress & Technical Documentation**

---

## 📌 1. Project Overview & Clinical Objective

### 🎯 The Problem
Hospital readmission within 30 days is a critical quality metric in healthcare (e.g., CMS Hospital Readmissions Reduction Program). Unplanned readmissions result in billions of dollars in excess healthcare costs and indicate potential gaps in discharge planning or post-acute care.

### 🎯 The Goal
Build an end-to-end Machine Learning system to predict **30-day hospital readmission (`<30` days)** for diabetic patients, enabling hospitals and clinical care teams to proactively intervene before patient discharge.

* **Dataset**: Diabetes 130-US Hospitals dataset (1999–2008)
* **Initial Size**: `101,766` encounters, `50` features
* **Cleaned Cohort**: `69,987` unique patients, `209` feature inputs
* **Primary Target**: `readmitted30_target` (`1` if readmitted `<30` days, `0` otherwise)

---

## 🔍 2. Exploratory Data Analysis (EDA) Findings

### A. Target Distribution & Class Imbalance
* **Target Classes**: `<30` days (~11.16% initially, ~8.98% after patient deduplication), `>30` days (~35%), `NO` (~54%).
* **Binary Formulation**: Patients readmitted in `<30` days are the high-risk positive class (`1`). The dataset has **significant class imbalance** (~8.98% positive class).

### B. Missing Value & Data Quality Patterns
* Missing values were recorded as string `'?'` instead of standard `NaN`.
* **`weight`**: ~97% missing `'?'` $\rightarrow$ dropped due to extreme missingness.
* **`payer_code`**: ~40% missing `'?'` (administrative insurance billing code with no clinical relevance) $\rightarrow$ dropped.
* **`medical_specialty`**: ~50% missing `'?'` $\rightarrow$ retained and imputed with `'Unknown'` (the absence of a specialist consult carries clinical meaning).
* **`race`**: ~2.2% missing `'?'` $\rightarrow$ imputed with `'Unknown'`.
* **`gender`**: 3 records had `'Unknown/Invalid'` $\rightarrow$ filtered out.

### C. Zero-Variance Columns
* **`examide`** and **`citoglipton`** had `nunique() == 1` (100% of rows were `"No"`) $\rightarrow$ dropped because they provide zero predictive signal.

### D. Critical Clinical Discoveries
1. **The Expired / Hospice Pattern in `discharge_disposition_id`**:
   - Discharge IDs `11` (1,642 patients), `19`, and `20` represent **Expired / Deceased** patients.
   - Discharge IDs `13`, `14` represent **Hospice** patients.
   - Deceased patients have a **`0.0%` readmission rate** by definition. Keeping them in the training set introduces severe target leakage and false bias.
2. **Prior Healthcare Utilization (`total_visits`)**:
   - `0` prior visits $\rightarrow$ **7.94%** readmission rate.
   - `1` prior visit $\rightarrow$ **10.60%** readmission rate.
   - `4` prior visits $\rightarrow$ **14.57%** readmission rate.
   - `11` prior visits $\rightarrow$ **27.08%** readmission rate ($>3\times$ higher risk!).
3. **Insulin Titration Risk**:
   - Dosage decreased (`Down`) $\rightarrow$ **13.90%** readmission rate (highest risk).
   - Dosage increased (`Up`) $\rightarrow$ **12.99%** readmission rate.
   - Stable dosage (`Steady`) $\rightarrow$ **11.13%** readmission rate.
   - No insulin (`No`) $\rightarrow$ **10.04%** readmission rate.
4. **Protective Effect of Metformin**:
   - Patients controlled on `Steady` Metformin had a lower readmission rate (**9.71%**) than patients taking no Metformin (**11.52%**).

---

## 🧹 3. Data Cleaning & Preprocessing Pipeline

| Step | Action Taken | Clinical / Technical Rationale |
| :--- | :--- | :--- |
| **1. Deceased / Hospice Filter** | Filtered out `discharge_disposition_id` in `[11, 13, 14, 19, 20, 21]` | Prevents target leakage (deceased patients cannot be readmitted). |
| **2. Invalid Gender Filter** | Removed 3 records where `gender == 'Unknown/Invalid'` | Eliminates corrupted data. |
| **3. Patient Deduplication** | Kept first encounter per `patient_nbr` (`drop_duplicates`) | Eliminates data leakage across train and test sets. |
| **4. Redundant Feature Drop** | Dropped `examide`, `citoglipton`, `weight`, `payer_code` | Eliminates zero-variance and >50% missing columns. |
| **5. Missing Value Imputation** | Replaced `'?'` with `'Unknown'` in `race` and `medical_specialty`; filled `NaN` with `'None'` in `A1Cresult` and `max_glu_serum` | Standardizes missing categorical representations. |

* **Resulting Cleaned Dataset Shape**: **`69,987` unique patient encounters**, **`47` columns**.

---

## 🧬 4. ICD-9 Diagnosis Grouping System

Raw diagnosis columns (`diag_1`, `diag_2`, `diag_3`) contained over **700+ specific ICD-9 codes**, creating high cardinality and noise. We mapped all codes into **9 standardized clinical disease categories**:

```
 700+ Raw ICD-9 Codes  ──►  map_icd9()  ──►  9 Standard Clinical Categories
```

| Clinical Category | ICD-9 Code Range / Rule | Clinical Description |
| :--- | :--- | :--- |
| **Circulatory** | `390–459` and `785` | Heart failure, coronary disease, hypertension |
| **Respiratory** | `460–519` and `786` | Pneumonia, bronchitis, asthma |
| **Digestive** | `520–579` and `787` | GI bleed, gastroenteritis |
| **Diabetes** | Starts with `'250'` (e.g. `250.01`, `250.6`) | Diabetic ketoacidosis, diabetes complications |
| **Injury** | `800–999` | Fractures, poisonings, acute trauma |
| **Musculoskeletal** | `710–739` | Osteoarthritis, joint disorders |
| **Genitourinary** | `580–629` and `788` | Kidney disease, urinary tract infections |
| **Neoplasms** | `140–239` | Malignant & benign tumors |
| **Other** | All other codes, `'V'`/`'E'` prefixes, `'?'` | Supplementary, screening, and rare codes |

---

## ⚙️ 5. Feature Engineering & Clinical Risk Indicators

We engineered 4 high-impact domain features:

1. **`total_visits`** (Total Prior Utilization):
   $$\text{total\_visits} = \text{number\_outpatient} + \text{number\_emergency} + \text{number\_inpatient}$$
2. **`num_diabetes_meds`** (Medication Burden / Complexity):
   $$\text{num\_diabetes\_meds} = \sum_{m \in \text{21 Medications}} \mathbb{I}(m \ne \text{'No'})$$
3. **`has_circulatory`** (Cardiovascular Comorbidity Flag):
   $$\text{has\_circulatory} = \mathbb{I}(\text{'Circulatory'} \in \{\text{diag\_1\_group}, \text{diag\_2\_group}, \text{diag\_3\_group}\})$$
4. **`has_diabetes_diag`** (Active Diabetes Complication Flag):
   $$\text{has\_diabetes\_diag} = \mathbb{I}(\text{'Diabetes'} \in \{\text{diag\_1\_group}, \text{diag\_2\_group}, \text{diag\_3\_group}\})$$

---

## 🔢 6. Categorical Encoding & Dataset Preparation

1. **Ordinal Age Encoding**: Mapped 10 age brackets to decade midpoints (`5, 15, 25, ..., 95`).
2. **Dropping Raw Columns**: Dropped `encounter_id`, `patient_nbr`, `diag_1`, `diag_2`, `diag_3`, `readmitted`.
3. **One-Hot Encoding**: Converted nominal variables via `pd.get_dummies(df, drop_first=True, dtype=int)`.
4. **Final Feature Matrix (`X`)**: **`69,987` rows $\times$ `209` numeric features**.

---

## 🤖 7. Model Benchmarking & Performance Comparison

| Model | ROC-AUC Score | Recall (Class 1) | Precision (Class 1) | Overall Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **1. Logistic Regression** | `0.6521` | `54%` | `14%` | `66%` |
| **2. Random Forest** | `0.6503` | `52%` | `15%` | `68%` |
| **3. XGBoost (Champion)** | **`0.6540`** 🥇 | `51.63%` | `14.41%` | **`68.11%`** |

* **Class Imbalance Strategy**: `scale_pos_weight = 10.14` applied to penalize positive class errors proportionally.
* **Top Predictive Features in XGBoost**: `total_visits`, `number_inpatient`, `discharge_disposition_id`, `insulin`, `time_in_hospital`, `num_lab_procedures`.

---

## 🏗️ 8. Production Modular Architecture (`src/`)

```
Healthcare-ML-System/
├── artifacts/
│   └── xgboost_metrics.json          # Exported evaluation metrics
├── models/
│   ├── xgboost_model.joblib          # Serialized trained model
│   └── feature_columns.json          # 209-column feature schema
├── src/
│   ├── ingestion/
│   │   └── ingest.py                 # DataIngestion class
│   ├── validation/
│   │   ├── schema.py                 # Required columns schema
│   │   └── validate.py               # DataValidator class
│   ├── preprocessing/
│   │   ├── clean.py                  # DataCleaner class
│   │   ├── missing.py                # MissingValueHandler class
│   │   └── encoding.py               # DataEncoder class
│   ├── feature_engineering/
│   │   └── create_features.py        # FeatureEngineer class
│   ├── training/
│   │   └── train.py                  # ModelTrainer class
│   ├── evaluation/
│   │   └── evaluate.py               # ModelEvaluator class
│   ├── pipeline/
│   │   └── training_pipeline.py      # End-to-end automated pipeline
│   └── api/
│       ├── schemas.py                # Pydantic PatientInput & Response schemas
│       └── main.py                   # FastAPI REST Prediction Endpoint
```

---

## ⚡ 9. CLI & Execution Commands

### 1. Run Automated Retraining Pipeline:
```bash
python -m src.pipeline.training_pipeline --model xgboost
```

### 2. Launch FastAPI Real-Time Scoring Service:
```bash
python -m src.api.main
```
* **Swagger UI Documentation**: `http://127.0.0.1:8000/docs`
* **Health Check**: `GET http://127.0.0.1:8000/health`
* **Predict Readmission**: `POST http://127.0.0.1:8000/predict`
