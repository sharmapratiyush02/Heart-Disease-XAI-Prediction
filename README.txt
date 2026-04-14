# Explainable AI (XAI) for Heart Disease Prediction

**B.Tech Seminar Project | MIT World Peace University, Pune**
**By: Pratiyush Sharma**
**Guide: Dr. Bharat Burgate**

## Overview
This project applies Explainable AI (XAI) techniques — SHAP and LIME — to
Machine Learning models for heart disease prediction using the UCI Heart Disease Dataset.

## Key Features
- ML Models: XGBoost (91% accuracy), Random Forest (88% accuracy)
- XAI: SHAP for global feature importance, LIME for patient-specific explanations
- Interactive Streamlit dashboard for clinical users
- Top predictors identified: Thalassemia, Chest Pain Type, Max Heart Rate, ST Depression

## Tech Stack
Python, scikit-learn, XGBoost, SHAP, LIME, Streamlit, Pandas, NumPy

## Dataset
UCI Heart Disease Dataset — 303 patients, 13 clinical features

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Results
| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| XGBoost | 91% | 0.96 |
| Random Forest | 88% | 0.93 |
| XGBoost + SHAP | 90% | 0.95 |
| Random Forest + LIME | 89% | 0.94 |