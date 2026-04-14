import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_utils import preprocess_and_train, get_shap_plot, get_lime_explanation

st.set_page_config(page_title="XAI Heart Disease Prediction", layout="wide")

st.title("🩺 Explainable AI (XAI) for Heart Disease Prediction")
st.markdown("This system uses **XGBoost** for prediction and **SHAP/LIME** for clinical transparency.")

# Load Model and Data
@st.cache_resource
def load_system():
    return preprocess_and_train()

model, scaler, X_train, X_test, X_train_scaled, X_test_scaled = load_system()
feature_names = X_train.columns.tolist()

# Sidebar - Patient Input
st.sidebar.header("Patient Clinical Parameters")

def user_input_features():
    # 1. Collect raw inputs
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
    
    # 2. Convert to DataFrame
    df_input = pd.DataFrame([data])
    
    # 3. Apply One-Hot Encoding (same as training)
    categorical_cols = ['cp', 'restecg', 'slope', 'thal']
    df_encoded = pd.get_dummies(df_input, columns=categorical_cols)
    
    # 4. THE FIX: Reindex to match the model's expected columns exactly
    # This adds missing columns (as 0) and removes extra columns
    df_final = df_encoded.reindex(columns=feature_names, fill_value=0)
    
    return df_final

# Get the processed input dataframe
input_df = user_input_features()

# --- MAIN PAGE ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Prediction Result")
    # Scale the input using the trained scaler
    patient_scaled = scaler.transform(input_df)
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0][1]
    
    if prediction == 1:
        st.error(f"🚨 Prediction: HEART DISEASE POSITIVE")
    else:
        st.success(f"✅ Prediction: HEART DISEASE NEGATIVE")
        
    st.metric("Confidence Score", f"{probability*100:.2f}%")

with col2:
    st.subheader("Local Explanation (LIME)")
    if st.button("Generate LIME Explanation"):
        # We pass the raw scaled values for LIME
        exp = get_lime_explanation(model, scaler, X_train_scaled, input_df.values[0], feature_names)
        fig_lime = exp.as_pyplot_figure()
        st.pyplot(fig_lime)
        st.info("LIME shows which specific features of THIS patient pushed the model toward the prediction.")

st.divider()

st.subheader("Global Model Insights (SHAP)")
if st.button("Show Global Feature Importance"):
    fig_shap = get_shap_plot(model, X_train_scaled, X_test_scaled, feature_names)
    st.pyplot(fig_shap)
    st.write("The SHAP Summary Plot shows the overall influence of each feature across the entire patient population.")