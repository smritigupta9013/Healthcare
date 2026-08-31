# 🏥 Healthcare ML System: Hospital Readmission Prediction

An end-to-end production Machine Learning system for predicting 30-day hospital readmissions in diabetic patients using the Diabetes 130-US Hospitals dataset.

---

## 📑 Documentation

* **Complete Technical Summary & Architecture**: [docs/PROJECT_PROGRESS.md](file:///c:/Users/Admin/Documents/python/Healthcare-ML-System/docs/PROJECT_PROGRESS.md)
* **Exploratory Data Analysis Notebook**: [notebooks/01_eda.ipynb](file:///c:/Users/Admin/Documents/python/Healthcare-ML-System/notebooks/01_eda.ipynb)

---

## 🚀 Key Achievements & Architecture

1. **Exploratory Data Analysis & Filtering**:
   - Analyzed 101,766 patient encounters, removed target leakage from deceased/hospice patients (IDs 11, 13, 14, 19, 20, 21), and deduplicated to `69,987` unique patients.
2. **Clinical Feature Engineering**:
   - Mapped 700+ raw ICD-9 codes to 9 clinical body system categories.
   - Built `total_visits`, `num_diabetes_meds`, `has_circulatory`, and `has_diabetes_diag`.
3. **Model Benchmarking**:
   - Logistic Regression (`0.6521` ROC-AUC)
   - Random Forest (`0.6503` ROC-AUC)
   - **XGBoost Champion (`0.6540` ROC-AUC, 51.63% Recall on 8.98% minority class)**.
4. **Modular Production Modules (`src/`)**:
   - Ingestion $\rightarrow$ Validation $\rightarrow$ Cleaning $\rightarrow$ Missing Imputation $\rightarrow$ Feature Engineering $\rightarrow$ Encoding $\rightarrow$ Training $\rightarrow$ Evaluation.
5. **Real-Time FastAPI Prediction Service**:
   - Real-time `/predict` endpoint with automated preprocessing, risk tiering, and actionable clinical guidance.

---

## ⚡ How to Run

### 1. Run Automated Training Pipeline
```bash
python -m src.pipeline.training_pipeline --model xgboost
```

### 2. Launch FastAPI Real-Time Scoring Service
```bash
python -m src.api.main
```
Open **Interactive Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
