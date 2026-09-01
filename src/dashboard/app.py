import sys
import os
import json
import joblib
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path so 'src' is importable when launched via streamlit run
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api.schemas import PatientInput
from src.preprocessing.missing import MissingValueHandler
from src.feature_engineering.create_features import FeatureEngineer, map_icd9
from src.preprocessing.encoding import DataEncoder

# Page configuration
st.set_page_config(
    page_title="Healthcare ML: Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clinical styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .risk-card-high {
        background-color: #FEE2E2;
        border-left: 6px solid #EF4444;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .risk-card-mod {
        background-color: #FEF3C7;
        border-left: 6px solid #F59E0B;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .risk-card-low {
        background-color: #D1FAE5;
        border-left: 6px solid #10B981;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_artifacts():
    """
    Loads trained model, feature columns, and evaluation metrics.
    """
    model_path = os.path.join(PROJECT_ROOT, "models", "xgboost_model.joblib")
    features_path = os.path.join(PROJECT_ROOT, "models", "feature_columns.json")
    metrics_path = os.path.join(PROJECT_ROOT, "artifacts", "xgboost_metrics.json")

    model = joblib.load(model_path) if os.path.exists(model_path) else None
    features = []
    if os.path.exists(features_path):
        with open(features_path, "r", encoding="utf-8") as f:
            features = json.load(f)

    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    return model, features, metrics


model, feature_columns, metrics = load_model_artifacts()


def predict_patient(patient_dict: dict):
    """
    Processes single patient dictionary and computes readmission probability.
    """
    df = pd.DataFrame([patient_dict])
    df = MissingValueHandler(df).handle_missing()
    df = FeatureEngineer(df).create_features()
    encoder = DataEncoder(df)
    df = encoder.encode_age()
    df = encoder.convert_id_cols_to_str()
    df = encoder.drop_raw_columns()
    df = encoder.one_hot_encode()
    df_aligned = df.reindex(columns=feature_columns, fill_value=0)

    prob = float(model.predict_proba(df_aligned)[0, 1])
    return prob


# Header
st.markdown('<div class="main-header">🏥 Hospital Readmission Risk Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Clinical Decision Support System for 30-Day Readmission Risk Stratification in Diabetic Inpatients</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital-3.png", width=70)
    st.title("Clinical AI Dashboard")
    st.markdown("---")

    if metrics:
        st.subheader("📊 Model Quality Metrics")
        st.metric("ROC-AUC Score", f"{metrics.get('roc_auc_score', 0.654):.3f}")
        st.metric("Readmission Recall", f"{metrics.get('class_1_readmitted', {}).get('recall', 0.516):.1%}")
        st.metric("Overall Accuracy", f"{metrics.get('overall_accuracy', 0.681):.1%}")
        st.caption(f"Tested on {metrics.get('test_patients_count', 13998):,} unseen patient records.")

    st.markdown("---")
    st.info("💡 **Clinical Tip**: Prior inpatient visits, insulin dosage reduction (`Down`), and cardiovascular diagnoses are top readmission risk drivers.")


# Main App Tabs
tab1, tab2, tab3 = st.tabs(["👤 Patient Risk Assessment", "📁 Batch Patient Stratification", "📈 Model Insights & Feature Importance"])

# TAB 1: Single Patient Assessment
with tab1:
    st.subheader("Enter Patient Discharge Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 1. Demographics & Admission")
        age = st.selectbox("Age Bracket", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"], index=7)
        gender = st.selectbox("Gender", ["Female", "Male"], index=0)
        race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Unknown"], index=0)
        time_in_hospital = st.slider("Time in Hospital (Days)", min_value=1, max_value=14, value=4)
        admission_type_id = st.selectbox("Admission Type ID", [1, 2, 3, 4, 5, 6, 7, 8], format_func=lambda x: {1: "1 - Emergency", 2: "2 - Urgent", 3: "3 - Elective"}.get(x, f"{x} - Other"), index=0)
        discharge_disposition_id = st.selectbox("Discharge Disposition ID", [1, 2, 3, 4, 5, 6, 7, 8, 18, 22, 25], format_func=lambda x: {1: "1 - Discharged to Home", 3: "3 - SNF / Skilled Nursing", 6: "6 - Home Health Care"}.get(x, f"{x} - Other"), index=0)
        admission_source_id = st.selectbox("Admission Source ID", [1, 2, 4, 7, 17], format_func=lambda x: {7: "7 - Emergency Room", 1: "1 - Physician Referral", 2: "2 - Clinic Referral"}.get(x, f"{x} - Other"), index=0)
        medical_specialty = st.selectbox("Medical Specialty", ["InternalMedicine", "Cardiology", "Family/GeneralPractice", "Surgery-General", "Emergency/Trauma", "Unknown"], index=0)

    with col2:
        st.markdown("#### 2. Prior Utilization & Labs")
        number_inpatient = st.number_input("Prior Inpatient Visits (Past Year)", min_value=0, max_value=20, value=1)
        number_emergency = st.number_input("Prior Emergency Visits (Past Year)", min_value=0, max_value=20, value=0)
        number_outpatient = st.number_input("Prior Outpatient Visits (Past Year)", min_value=0, max_value=20, value=0)
        num_medications = st.slider("Number of Medications Prescribed", min_value=1, max_value=60, value=15)
        num_lab_procedures = st.slider("Number of Lab Procedures Performed", min_value=1, max_value=120, value=45)
        num_procedures = st.slider("Number of Surgical Procedures", min_value=0, max_value=6, value=1)
        number_diagnoses = st.slider("Total Number of Diagnoses Recorded", min_value=1, max_value=16, value=7)
        A1Cresult = st.selectbox("HbA1c Lab Test Result", ["None", "Norm", ">7", ">8"], index=0)
        max_glu_serum = st.selectbox("Max Glucose Serum Result", ["None", "Norm", ">200", ">300"], index=0)

    with col3:
        st.markdown("#### 3. Diagnoses & Medications")
        diag_1 = st.text_input("Primary Diagnosis ICD-9 Code", value="428", help="e.g. 428 for Heart Failure, 250 for Diabetes")
        diag_2 = st.text_input("Secondary Diagnosis ICD-9 Code", value="250.01", help="e.g. 250.01 for Diabetes, 401 for Hypertension")
        diag_3 = st.text_input("Tertiary Diagnosis ICD-9 Code", value="401")
        insulin = st.selectbox("Insulin Dosage Titration", ["No", "Steady", "Up", "Down"], index=3, help="Note: Down indicates dosage was decreased during stay.")
        metformin = st.selectbox("Metformin Titration", ["No", "Steady", "Up", "Down"], index=1)
        glipizide = st.selectbox("Glipizide", ["No", "Steady", "Up", "Down"], index=0)
        glyburide = st.selectbox("Glyburide", ["No", "Steady", "Up", "Down"], index=0)
        pioglitazone = st.selectbox("Pioglitazone", ["No", "Steady", "Up", "Down"], index=0)
        change = st.selectbox("Change in Diabetic Medication", ["Ch", "No"], index=0)
        diabetesMed = st.selectbox("Any Diabetes Medication Prescribed?", ["Yes", "No"], index=0)

    st.markdown("---")

    if st.button("⚡ Calculate Readmission Risk Score", type="primary", use_container_width=True):
        if model is None:
            st.error("Model artifacts not found! Run the training pipeline first: `python -m src.pipeline.training_pipeline`")
        else:
            patient_data = {
                "race": race,
                "gender": gender,
                "age": age,
                "admission_type_id": admission_type_id,
                "discharge_disposition_id": discharge_disposition_id,
                "admission_source_id": admission_source_id,
                "time_in_hospital": time_in_hospital,
                "medical_specialty": medical_specialty,
                "num_lab_procedures": num_lab_procedures,
                "num_procedures": num_procedures,
                "num_medications": num_medications,
                "number_outpatient": number_outpatient,
                "number_emergency": number_emergency,
                "number_inpatient": number_inpatient,
                "diag_1": diag_1,
                "diag_2": diag_2,
                "diag_3": diag_3,
                "number_diagnoses": number_diagnoses,
                "max_glu_serum": max_glu_serum,
                "A1Cresult": A1Cresult,
                "metformin": metformin,
                "repaglinide": "No", "nateglinide": "No", "chlorpropamide": "No", "glimepiride": "No",
                "acetohexamide": "No", "glipizide": glipizide, "glyburide": glyburide, "tolbutamide": "No",
                "pioglitazone": pioglitazone, "rosiglitazone": "No", "acarbose": "No", "miglitol": "No",
                "troglitazone": "No", "tolazamide": "No", "insulin": insulin,
                "glyburide-metformin": "No", "glipizide-metformin": "No", "glimepiride-pioglitazone": "No",
                "metformin-rosiglitazone": "No", "metformin-pioglitazone": "No",
                "change": change,
                "diabetesMed": diabetesMed,
            }

            prob = predict_patient(patient_data)

            st.markdown("### 📋 Clinical Assessment Results")
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                st.metric(
                    label="30-Day Readmission Probability",
                    value=f"{prob * 100:.1f}%",
                    delta="High Risk" if prob >= 0.50 else ("Moderate Risk" if prob >= 0.25 else "Low Risk"),
                    delta_color="inverse" if prob >= 0.25 else "normal",
                )

            with res_col2:
                if prob >= 0.50:
                    st.markdown("""
                    <div class="risk-card-high">
                        <h4 style="color: #991B1B; margin-top:0;">🔴 HIGH READMISSION RISK DETECTED</h4>
                        <p><strong>Recommended Post-Discharge Action:</strong></p>
                        <ul>
                            <li>Schedule Home Health Care or Nurse phone follow-up within <strong>48 hours</strong>.</li>
                            <li>Perform pharmacist medication reconciliation (titration changes detected).</li>
                            <li>Ensure primary care appointment within <strong>7 days</strong>.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                elif prob >= 0.25:
                    st.markdown("""
                    <div class="risk-card-mod">
                        <h4 style="color: #92400E; margin-top:0;">🟡 MODERATE READMISSION RISK</h4>
                        <p><strong>Recommended Action:</strong> Coordinate care management check-in call within 7 days and verify patient comprehension of discharge medications.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="risk-card-low">
                        <h4 style="color: #065F46; margin-top:0;">🟢 LOW READMISSION RISK</h4>
                        <p><strong>Recommended Action:</strong> Standard post-discharge recovery instructions and routine primary care follow-up.</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Clinical Insights Breakdown
            st.markdown("#### 🔍 Clinical Risk Drivers Identified for this Patient:")
            insights = []
            if number_inpatient > 0:
                insights.append(f"• **Prior Inpatient Utilization**: Patient has **{number_inpatient}** prior inpatient visit(s), strongly escalating readmission risk.")
            if insulin == "Down":
                insights.append("• **Insulin Dosage Reduced (`Down`)**: Clinical titration reduction is the highest-risk medication adjustment pattern.")
            if map_icd9(diag_1) == "Circulatory" or map_icd9(diag_2) == "Circulatory":
                insights.append("• **Cardiovascular Comorbidity**: Patient has active circulatory/cardiovascular disease diagnoses.")
            if num_medications > 15:
                insights.append(f"• **Polypharmacy**: Patient is prescribed **{num_medications}** medications simultaneously.")

            if insights:
                for ins in insights:
                    st.write(ins)
            else:
                st.write("• No severe risk flags detected. Patient exhibits stable clinical indicators.")


# TAB 2: Batch Stratification
with tab2:
    st.subheader("Batch Patient Cohort Risk Stratification")
    st.write("Upload a CSV file containing patient encounters to score and categorize an entire hospital unit.")

    uploaded_file = st.file_uploader("Upload Patient CSV", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Uploaded **{len(batch_df):,}** patient records.")

        if st.button("🚀 Process & Score Cohort"):
            with st.spinner("Scoring patient cohort with XGBoost..."):
                probs = []
                for _, row in batch_df.iterrows():
                    p_dict = row.to_dict()
                    p_prob = predict_patient(p_dict)
                    probs.append(p_prob)

                batch_df["readmission_risk_prob"] = [round(p, 4) for p in probs]
                batch_df["risk_tier"] = [
                    "High Risk" if p >= 0.50 else ("Moderate Risk" if p >= 0.25 else "Low Risk")
                    for p in probs
                ]

                st.success("Batch scoring complete!")
                st.dataframe(batch_df[["patient_nbr" if "patient_nbr" in batch_df.columns else "encounter_id", "age", "time_in_hospital", "readmission_risk_prob", "risk_tier"]].head(20))

                csv_data = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Scored Patients CSV",
                    data=csv_data,
                    file_name="scored_patient_cohort.csv",
                    mime="text/csv",
                )


# TAB 3: Model Insights
with tab3:
    st.subheader("Model Performance & Feature Importance")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Top 10 Most Predictive Features (XGBoost)")
        if model is not None and feature_columns:
            feat_imp = pd.Series(model.feature_importances_, index=feature_columns).nlargest(10)
            st.bar_chart(feat_imp)

    with col_b:
        st.markdown("#### Key Clinical Interpretations")
        st.write("""
        1. **Total Prior Visits (`total_visits` & `number_inpatient`)**: Patients with frequent previous hospitalizations carry over **3x higher** readmission likelihood.
        2. **Insulin Titration (`insulin_Down`)**: Decreasing insulin dosages during acute inpatient stays correlates with glycemic instability post-discharge.
        3. **Cardiovascular Diagnoses (`has_circulatory`)**: Heart failure and hypertensive complications form the largest comorbidity block.
        4. **Length of Stay (`time_in_hospital`)**: Longer hospitalizations reflect clinical complexity and higher frailty.
        """)
