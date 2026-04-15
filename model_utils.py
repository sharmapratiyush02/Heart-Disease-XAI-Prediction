import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

def preprocess_and_train():
    df = pd.read_csv('heart_disease.csv')
    df['ca'] = df['ca'].fillna(df['ca'].median())
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0])
    categorical_cols = ['cp', 'restecg', 'slope', 'thal']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='logloss')
    model.fit(X_train_scaled, y_train)
    return model, scaler, X_train, X_test, X_train_scaled, X_test_scaled

def get_shap_global_plot(model, X_train_scaled, X_test_scaled, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, show=False)
    plt.tight_layout()
    return fig

def get_shap_local_plot(model, scaler, patient_data, feature_names):
    explainer = shap.TreeExplainer(model)
    patient_scaled = scaler.transform([patient_data])
    shap_values = explainer.shap_values(patient_scaled)
    
    # Create a bar chart for local SHAP values
    fig, ax = plt.subplots(figsize=(10, 6))
    # We use the first (and only) patient's values
    shap.bar_plot(shap_values[0], feature_names=feature_names, show=False)
    plt.title("Local SHAP: Contribution to Prediction")
    plt.tight_layout()
    return fig

def get_lime_explanation(model, scaler, X_train_scaled, patient_data, feature_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_scaled,
        feature_names=feature_names,
        class_names=['No Disease', 'Heart Disease'],
        mode='classification'
    )
    patient_scaled = scaler.transform([patient_data])
    exp = explainer.explain_instance(patient_scaled[0], model.predict_proba, num_features=8)
    return exp