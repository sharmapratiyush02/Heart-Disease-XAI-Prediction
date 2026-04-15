import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_utils import preprocess_and_train, get_shap_global_plot, get_shap_local_plot, get_lime_explanation

st.set_page_config(page_title="Clinical XAI Dashboard", layout="wide")

# --- HIGH CONTRAST CSS (Works for Dark & Light Mode) ---
st.markdown("""
    <style>
    .reasoning-box {
        background-color: rgba(255, 75, 75, 0.1);
        border-left: 5px solid #ff4b4b;
        padding: 15px;
        border-radius: 5px;
        color: #ff4b4b; 
        border: 1px solid #ff4b4b;
    }
    .patient-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid #4b4b4b;
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #aaa;
        font-weight: bold;
    }
    .metric-value {
        font-size: 1.1rem;
        color: #fff;
        font-weight: 500;
    }
    .xai-step {
        background-color: rgba(2, 136, 209, 0.2);
        border-left: 5px solid #0288d1;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        color: #00d4ff !important; 
        border: 1px solid #0288d1;
        font-weight: 500;
    }
    .interpretation-guide {
        background-color: rgba(0, 0, 0, 0.3);
        padding: 10px;
        border-radius: 5px;
        border: 1px dashed #666;
        color: #ddd;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Clinical Decision Support System with XAI")
st.markdown("### Bridging the gap between Black-Box AI and Medical Trust")

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
    return df_final, data

input_df, raw_data = user_input_features()

# --- TOP SECTION: THE CLINICAL DASHBOARD ---
col1, col2, col3 = st.columns([1, 1.2, 1])

with col1:
    st.subheader("AI Diagnosis")
    patient_scaled = scaler.transform(input_df)
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0][1]
    
    if prediction == 1:
        st.error(f"🚨 POSITIVE for Heart Disease")
    else:
        st.success(f"✅ NEGATIVE for Heart Disease")
    st.metric("Model Confidence", f"{probability*100:.2f}%")

with col2:
    st.subheader("Patient Snapshot")
    snapshot_html = f'''
    <div class="patient-card">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div><p class="metric-label">AGE</p><p class="metric-value">{raw_data['age']} yrs</p></div>
            <div><p class="metric-label">SEX</p><p class="metric-value">{'Male' if raw_data['sex']==1 else 'Female'}</p></div>
            <div><p class="metric-label">CHEST PAIN</p><p class="metric-value">Type {raw_data['cp']}</p></div>
            <div><p class="metric-label">BP (Resting)</p><p class="metric-value">{raw_data['trestbps']} mmHg</p></div>
            <div><p class="metric-label">CHOLESTEROL</p><p class="metric-value">{raw_data['chol']} mg/dl</p></div>
            <div><p class="metric-label">MAX HR</p><p class="metric-value">{raw_data['thalach']} bpm</p></div>
            <div><p class="metric-label">ST DEPRESSION</p><p class="metric-value">{raw_data['oldpeak']}</p></div>
            <div><p class="metric-label">MAJOR VESSELS</p><p class="metric-value">{raw_data['ca']}</p></div>
        </div>
    </div>
    '''
    st.markdown(snapshot_html, unsafe_allow_html=True)

with col3:
    st.subheader("XAI Logic")
    st.markdown(f'''
    <div class="reasoning-box">
        <b>🧠 Clinical Reasoning:</b><br>
        The AI is <b>{probability*100:.1f}%</b> confident. However, a probability is not a diagnosis. 
        We must now verify if the <b>{raw_data['age']}yo</b> patient's results correlate with 
        known clinical markers using SHAP and LIME.
    </div>
    ''', unsafe_allow_html=True)

st.divider()

# --- XAI WORKFLOW SECTION ---
st.subheader("🔍 Explainability Analysis")
workflow = st.radio(
    "Select Analysis Workflow:",
    ("Parallel Validation (SHAP & LIME Together)", "Sequential Discovery (Global → Local → Deep Dive)"),
    horizontal=True
)

if workflow == "Parallel Validation (SHAP & LIME Together)":
    st.markdown('''<div class="xai-step"><b>🎯 Objective:</b> Cross-verify two different XAI mathematics. If both SHAP and LIME point to the same feature, the clinical confidence increases.</div>''', unsafe_allow_html=True)
    
    col_shap, col_lime = st.columns(2)
    with col_shap:
        st.markdown("#### 🟢 SHAP (Game Theory)")
        st.pyplot(get_shap_local_plot(model, scaler, input_df.values[0], feature_names))
        st.markdown('''<div class="interpretation-guide"><b>Interpretation:</b> Bars to the <b>Right</b> increase risk; bars to the <b>Left</b> decrease it. SHAP tells us the exact "contribution weight" of this patient's vitals.</div>''', unsafe_allow_html=True)
        
    with col_lime:
        st.markdown("#### 🟡 LIME (Local Surrogate)")
        exp = get_lime_explanation(model, scaler, X_train_scaled, input_df.values[0], feature_names)
        st.pyplot(exp.as_pyplot_figure())
        st.markdown('''<div class="interpretation-guide"><b>Interpretation:</b> LIME creates a "Simplified Local Rule". It tells us: "Because this specific value is X, the risk increases by Y%". It is the most human-readable explanation.</div>''', unsafe_allow_html=True)

elif workflow == "Sequential Discovery (Global → Local → Deep Dive)":
    st.markdown('''<div class="xai-step"><b>🎯 Objective:</b> Follow a clinical logic path: General Model Trends → Individual Patient Impact → Local Feature Trigger.</div>''', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["1. Global Knowledge", "2. Patient Impact", "3. Clinical Deep Dive"])
    with tab1:
        st.markdown("#### Step 1: What the Model Learned (Global)")
        st.pyplot(get_shap_global_plot(model, X_train_scaled, X_test_scaled, feature_names))
        st.markdown('''<div class="interpretation-guide"><b>Clinical Logic:</b> We first look at the entire population. This chart shows which features the AI considers most important <i>in general</i>. If 'Thalassemia' is at the top, the AI is primarily a 'Blood Disorder' detector.</div>''', unsafe_allow_html=True)
        
    with tab2:
        st.markdown("#### Step 2: How it applied to THIS Patient (Local SHAP)")
        st.pyplot(get_shap_local_plot(model, scaler, input_df.values[0], feature_names))
        st.markdown('''<div class="interpretation-guide"><b>Clinical Logic:</b> Now we isolate this patient. We check if the "Global" importance matches the "Local" importance. If the patient has high cholesterol but it's not pushing the risk up, the AI is ignoring it for this specific case.</div>''', unsafe_allow_html=True)
        
    with tab3:
        st.markdown("#### Step 3: Local Sensitivity Analysis (LIME)")
        exp = get_lime_explanation(model, scaler, X_train_scaled, input_df.values[0], feature_names)
        st.pyplot(exp.as_pyplot_figure())
        st.markdown('''<div class="interpretation-guide"><b>Clinical Logic:</b> Finally, we find the "Tipping Point". LIME tells the doctor: "If this patient's Max Heart Rate was 10bpm higher, the diagnosis would have flipped to Negative." This is actionable medical insight.</div>''', unsafe_allow_html=True)