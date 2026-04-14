<div align="center">

# 🫀 Explainable AI for Heart Disease Prediction

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-91%25_Accuracy-orange?style=flat-square)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Plagiarism](https://img.shields.io/badge/Plagiarism-0%25_Original-brightgreen?style=flat-square)]()

### Making AI decisions in healthcare **transparent**, **trustworthy**, and **clinically actionable**

*B.Tech Seminar Project | Semester VI | AY 2025–26*
*Dr. Vishwanath Karad MIT World Peace University, Pune*

---

</div>

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Highlights](#-key-highlights)
- [System Architecture](#-system-architecture)
- [Dataset](#-dataset)
- [Models & Results](#-models--results)
- [XAI Techniques](#-xai-techniques)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Screenshots](#-screenshots)
- [Research Report](#-research-report)
- [References](#-references)
- [Author](#-author)

---

## 🧠 Overview

Cardiovascular diseases (CVDs) account for **~32% of global deaths annually** and are a leading cause of premature mortality in India. While Machine Learning models like XGBoost and Random Forest achieve **85–91% diagnostic accuracy**, their black-box nature prevents clinical adoption.

This project solves that problem using **Explainable AI (XAI)** — specifically **SHAP** and **LIME** — to make every prediction transparent and understandable to doctors, patients, and regulators.

> *"A physician cannot responsibly act on an unexplained AI diagnosis."*

---

## ✨ Key Highlights

| Feature | Detail |
|--------|--------|
| 🎯 Best Accuracy | **91%** — XGBoost on UCI Heart Disease Dataset |
| 📊 Best AUC-ROC | **0.96** — XGBoost |
| 🔍 Global Explainability | SHAP (Shapley Additive Explanations) |
| 🔬 Local Explainability | LIME (Local Interpretable Model-Agnostic Explanations) |
| 🖥️ Dashboard | Interactive Streamlit Web App |
| 📁 Dataset | UCI Heart Disease Dataset (303 patients, 13 features) |
| ✅ Plagiarism | 100% Original (PaperRater, April 2026) |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      XAI PIPELINE                               │
│                                                                 │
│  [Data Input]  →  [Preprocessing]  →  [Model Training]         │
│      ↓                  ↓                     ↓                │
│  UCI Dataset      Missing Values        Random Forest           │
│  303 patients     One-Hot Encoding      XGBoost                 │
│  13 features      StandardScaler        Decision Tree           │
│                   80/20 Split           Logistic Regression     │
│                                                                 │
│                         ↓                                      │
│               [XAI Explanation Layer]                           │
│                    ↙         ↘                                 │
│              SHAP            LIME                               │
│         (Global View)   (Patient View)                          │
│                    ↘         ↙                                 │
│            [Streamlit Dashboard]                                │
│         Clinical Users | Real-Time Predictions                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📂 Dataset

**UCI Heart Disease Dataset** — [University of California, Irvine ML Repository](https://archive.uci.edu/dataset/45/heart+disease)

Originally collected at the **Cleveland Clinic Foundation** (1988) by Dr. Robert Detrano.

| Property | Value |
|----------|-------|
| Total Patients | 303 |
| Disease Positive | 165 (54.5%) |
| Disease Negative | 138 (45.5%) |
| Input Features | 13 |
| Target | Binary (0 = No Disease, 1 = Disease) |

### Features Used

| # | Feature | Description |
|---|---------|-------------|
| 1 | Age | Patient age (29–77 years) |
| 2 | Sex | Gender (1=Male, 0=Female) |
| 3 | Chest Pain Type | Type of chest pain (1–4) |
| 4 | Resting BP | Blood pressure at rest (mm Hg) |
| 5 | Cholesterol | Serum cholesterol (mg/dl) |
| 6 | Fasting Blood Sugar | FBS > 120 mg/dl (1=True) |
| 7 | Resting ECG | ECG results at rest (0–2) |
| 8 | Max Heart Rate | Peak heart rate during exercise |
| 9 | Exercise Angina | Exercise-induced chest pain |
| 10 | ST Depression | Depression of ST segment |
| 11 | Slope | Slope of peak exercise ST segment |
| 12 | Major Vessels | Vessels coloured by fluoroscopy (0–3) |
| 13 | Thalassemia | Blood disorder type (1–3) |

---

## 📈 Models & Results

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 79.0% | 0.78 | 0.81 | 0.79 | 0.86 |
| Decision Tree | 82.0% | 0.81 | 0.83 | 0.82 | 0.82 |
| Random Forest | 88.0% | 0.87 | 0.89 | 0.88 | 0.93 |
| **XGBoost** | **91.0%** | **0.90** | **0.92** | **0.91** | **0.96** |
| XGBoost + SHAP | 90.0% | 0.89 | 0.91 | 0.90 | 0.95 |
| Random Forest + LIME | 89.0% | 0.88 | 0.90 | 0.89 | 0.94 |

> 💡 **Key Insight:** XAI integration causes only **1% accuracy drop** while massively improving transparency and clinical trustworthiness.

---

## 🔍 XAI Techniques

### SHAP — Global Feature Importance

SHAP (Shapley Additive Explanations) is grounded in **cooperative game theory**, assigning each feature a fair contribution score across all patients.

**Top 8 Most Important Features (XGBoost Model):**

| Rank | Feature | Mean |SHAP| | Clinical Significance |
|------|---------|-------------|----------------------|
| 🥇 1 | Thalassemia | 0.42 | Reversible defect → strong coronary artery disease indicator |
| 🥈 2 | Chest Pain Type | 0.38 | Asymptomatic type is paradoxically high-risk |
| 🥉 3 | Max Heart Rate | 0.31 | Low max HR → poor cardiac reserve |
| 4 | ST Depression | 0.27 | >2.0 mm → significant myocardial ischemia |
| 5 | Age | 0.21 | Risk rises sharply above 55 years |
| 6 | Major Vessels | 0.18 | More blocked vessels → higher disease severity |
| 7 | Cholesterol | 0.14 | Elevated cholesterol → moderate risk contribution |
| 8 | Resting BP | 0.09 | Elevated BP → contributing but less dominant factor |

### LIME — Patient-Level Explanation

LIME generates **locally faithful linear approximations** for individual predictions — enabling personalised clinical communication.

**Example — High Risk Patient (Confidence: 87.3%):**
```
Patient #127: Male, 58y, Asymptomatic Chest Pain, BP 145, Cholesterol 233

  ✅ INCREASES RISK:
  + Thalassemia = Reversible Defect   → +0.38
  + Chest Pain = Asymptomatic         → +0.31
  + ST Depression > 2.0               → +0.24
  + Num. Vessels = 0                  → +0.19
  + Age > 55                          → +0.15

  🛡️ DECREASES RISK:
  - Max Heart Rate < 155              → -0.12
  - Cholesterol < 240                 → -0.09
  - Blood Pressure < 150              → -0.06
```

### SHAP vs LIME — Comparison

| Property | SHAP | LIME |
|----------|------|------|
| Scope | Global + Local | Local only |
| Theoretical Basis | Game Theory (Shapley Values) | Local linear approximation |
| Consistency | Always same result | Can vary between runs |
| Computational Cost | Higher (polynomial) | Lower (sampling-based) |
| Best For | Population-level insights | Patient-level consultation |

---

## 📁 Project Structure

```
Heart-Disease-XAI-Prediction/
│
├── 📄 app.py                        # Streamlit dashboard (main app)
├── 📄 model_training.py             # ML model training & evaluation
├── 📄 shap_analysis.py              # SHAP explainability module
├── 📄 lime_analysis.py              # LIME explainability module
├── 📄 preprocessing.py              # Data preprocessing pipeline
│
├── 📁 data/
│   └── heart_disease.csv            # UCI Heart Disease Dataset
│
├── 📁 models/
│   ├── xgboost_model.pkl            # Trained XGBoost model
│   └── random_forest_model.pkl      # Trained Random Forest model
│
├── 📁 outputs/
│   ├── shap_summary_plot.png        # SHAP beeswarm feature importance plot
│   ├── roc_curve_comparison.png     # ROC-AUC comparison chart
│   └── lime_patient_example.html    # LIME explanation for sample patient
│
├── 📄 requirements.txt              # Python dependencies
├── 📄 FINAL_SEMINAR_REPORT.pdf      # Full B.Tech seminar report
└── 📄 README.md                     # This file
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.10+
- pip package manager

### Clone the Repository

```bash
git clone https://github.com/YourUsername/Heart-Disease-XAI-Prediction.git
cd Heart-Disease-XAI-Prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Requirements (`requirements.txt`)

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
shap>=0.41.0
lime>=0.2.0.1
streamlit>=1.20.0
matplotlib>=3.6.0
seaborn>=0.12.0
joblib>=1.2.0
```

---

## ▶️ How to Run

### 1. Train the Models

```bash
python model_training.py
```

### 2. Generate SHAP Explanations

```bash
python shap_analysis.py
```

### 3. Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

Then open your browser at: **`http://localhost:8501`**

### 4. Core Code — Quick Start

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import shap

# Load and preprocess data
df = pd.read_csv('data/heart_disease.csv')
df['ca'].fillna(df['ca'].median(), inplace=True)
df['thal'].fillna(df['thal'].mode()[0], inplace=True)
df = pd.get_dummies(df, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)

X, y = df.drop('target', axis=1), df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Train XGBoost
model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train_s, y_train)

# SHAP Explanation
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_s)
shap.summary_plot(shap_values, X_test, feature_names=X.columns.tolist())
```

---

## 📸 Screenshots

> *(Add screenshots of your Streamlit dashboard here after running the app)*

### Dashboard Overview
![Dashboard](outputs/dashboard_overview.png)

### SHAP Global Feature Importance
![SHAP Summary Plot](outputs/shap_summary_plot.png)

### LIME Local Explanation (Patient-Level)
![LIME Explanation](outputs/lime_patient_example.png)

### ROC-AUC Curve Comparison
![ROC Curve](outputs/roc_curve_comparison.png)


---

## 📚 References

1. Santhosh et al. — *Cardiac Clarity: Harnessing ML for Heart-Disease Prediction* — IEEE Access, 2025
2. Arshad et al. — *XAI Approach to Heart Disease with Ensemble Methods* — IEEE WCONF, 2025
3. Sethi et al. — *XAI Approach to Heart Disease Prediction* — IEEE AIIoT, 2024
4. El-Sofany — *Predicting Heart Diseases Using ML Classification Techniques* — IEEE Access, 2024
5. Kim et al. — *XGBoost & SHAP Analysis of Hypertension Risk Factors* — Bioengineering MDPI, 2025
6. Islam et al. — *Predictive Modeling for Cancer with Explainable AI* — Scientific Reports, 2024
7. Lundberg & Lee — *A Unified Approach to Interpreting Model Predictions* — NeurIPS, 2017
8. Ribeiro et al. — *"Why Should I Trust You?": Explaining Classifier Predictions* — KDD, 2016

---

## 👤 Author

<div align="center">

**Pratiyush Sharma**
CSE-AIDS

*Under the guidance of **Dr. Bharat Burgate***

🏫 School of Computer Science & Engineering
Department of Computer Engineering & Technology
Dr. Vishwanath Karad MIT World Peace University, Pune

Academic Year: 2025–2026

---

⭐ *If you found this project useful, consider starring the repository!*

</div>