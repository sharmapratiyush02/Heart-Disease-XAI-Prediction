import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_utils import preprocess_and_train, get_shap_global_plot, get_shap_local_plot, get_lime_explanation

st.set_page_config(page_title="Clinical XAI Dashboard", layout="wide")

# --- CUSTOM CSS FOR REASONING BOXES ---
st.markdown("""
    <style>
    .reasoning-box {
        background-color: #f0f2f6;
        border-left: 5px solid #ff4b4b;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .xai-step {
        background-color: #e1f5fe;
        border-left: 5px solid #0288d1;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Clinical Decision Support System with XAI")
st.markdown("### Bridging the gap between Black-Box AI and Medical Trust")

# Load System
@st.cache_resource
def load_system():
    return preprocess_and_train()

model, scaler, X_train, X_test, X_train_scaled, X_test_scaled = load_system()
feature_names = X_train.columns.tolist()

# --- SIDEBAR INPUTS ---
st.sidebar.header("🏥 Patient Vital Signs")
def user_input_features():
    data = {
        'age': st.sidebar.slider("Age", 29, 77, 50),
        'sex': st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x==1 else "Female"),
        'cp': st.sidebar.selectbox("Chest Pain Type", options=[1, 2, 3, 4]),
        'trestbps': st.sidebar.slider("Resting Blood Pressure", 94, 200, 120),
        'chol': st.sidebar.slider("Cholesterol", 126, 564, 200),
        'fbs': st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "True" if x==1 else "False"),
        'restecg': st.sidebar.selectbox("Resting ECG", options=[0, 1, 2]),
        'thalach': st.sidebar.slider("Max Heart Rate", 71, 202, 150),
        'exang': st.sidebar.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No"),
        'oldpeak': st.sidebar.slider("ST Depression", 0.0, 6.2, 1.0),
        'slope': st.sidebar.selectbox("Slope", options=[1, 2, 3]),
        'ca': st.sidebar.slider("Num Major Vessels", 0, 3, 0),
        'thal': st.sidebar.selectbox("Thalassemia", options=[1, 2, 3])
    }
    df_input = pd.DataFrame([data])
    categorical_cols = ['cp', 'restecg', 'slope', 'thal']
    df_encoded = pd.get_dummies(df_input, columns=categorical_cols)
    df_final = df_encoded.reindex(columns=feature_names, fill_value=0)
    return df_final

input_df = user_input_features()

# --- PREDICTION SECTION ---
col_res1, col_res2 = st.columns([1, 2])
with col_res1:
    st.subheader("AI Diagnosis")
    patient_scaled = scaler.transform(input_df)
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0][1]
    
    if prediction == 1:
        st.error(f"🚨 POSITIVE for Heart Disease")
    else:
        st.success(f"✅ NEGATIVE for Heart Disease")
    st.metric("Model Confidence", f"{probability*100:.2f}%")

with col_res2:
    st.markdown('''<div class="reasoning-box"><b>🧠 XAI Reasoning:</b> Why can't we stop here? A confidence score of 90% tells us the AI is sure, but not <i>why</i>. In medicine, "The AI said so" is not a valid clinical justification. We use SHAP and LIME to audit this specific decision.</div>''', unsafe_allow_html=True)
st.divider()

# --- XAI WORKFLOW SELECTION ---
st.subheader("🔍 Explainability Analysis")
workflow = st.radio(
    "Select Analysis Workflow:",
    ("Parallel Validation (SHAP & LIME Together)", "Sequential Discovery (Global → Local → Deep Dive)"),
    help="Parallel is for verifying consistency; Sequential is for clinical diagnostic reasoning."
)

if workflow == "Parallel Validation (SHAP & LIME Together)":
    st.markdown('''<div class="xai-step"><b>Objective:</b> Compare two different XAI mathematics on the same data. If both SHAP and LIME point to 'Thalassemia' as the cause, the clinical confidence increases.</div>''', unsafe_allow_html=True)
    
    col_shap, col_lime = st.columns(2)
    
    with col_shap:
        st.markdown("#### 🟢 SHAP (Game Theory)")
        st.caption("Calculates the exact contribution of each feature to the final score.")
        fig_shap = get_shap_local_plot(model, scaler, input_df.values[0], feature_names)
        st.pyplot(fig_shap)
        
    with col_lime:
        st.markdown("#### 🟡 LIME (Local Surrogate)")
        st.caption("Perturbs the data to see which features, if changed, would flip the prediction.")
        exp = get_lime_explanation(model, scaler, X_train_scaled, input_df.values[0], feature_names)
        fig_lime = exp.as_pyplot_figure()
        st.pyplot(fig_lime)

elif workflow == "Sequential Discovery (Global → Local → Deep Dive)":
    st.markdown('<div class="xai-step"><b>Objective:</b> Follow a clinical logic path: What does the model usually look for? → What did it see in this patient? → How does this specific feature act as a trigger?</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["1. Global Knowledge", "2. Patient Impact", "3. Clinical Deep Dive"])
    
    with tab1:
        st.markdown("#### Step 1: What the Model Learned (Global)")
        st.write("Before looking at the patient, we see what the XGBoost model values most across 303 patients.")
        fig_global = get_shap_global_plot(model, X_train_scaled, X_test_scaled, feature_names)
        st.pyplot(fig_global)
        st.info("💡 **Reasoning:** If 'Thalassemia' is the top global feature, the doctor knows the model is biased toward blood disorder indicators.")

    with tab2:
        st.markdown("#### Step 2: How it applied to THIS Patient (Local SHAP)")
        st.write("Now we apply that global knowledge to the specific input. Which features pushed the probability up or down?")
        fig_local = get_shap_local_plot(model, scaler, input_df.values[0], feature_names)
        st.pyplot(fig_local)
        st.info("💡 **Reasoning:** Here we see the 'Why'. E.g., 'Patient's high age pushed the risk up, but their low cholesterol pulled it down.'")

    with tab3:
        st.markdown("#### Step 3: Local Sensitivity Analysis (LIME)")
        st.write("Finally, we use LIME to see the 'Decision Boundary'. If we changed the Chest Pain type, would the diagnosis change?")
        exp = get_lime_explanation(model, scaler, X_train_scaled, input_df.values[0], feature_names)
        fig_lime = exp.as_pyplot_figure()
        st.pyplot(fig_lime)
        st.info("💡 **Reasoning:** LIME provides a 'human-readable' rule (e.g., 'If Age > 55, risk increases'). This is the most actionable part for a physician.")