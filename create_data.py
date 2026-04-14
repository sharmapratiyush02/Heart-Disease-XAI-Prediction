import pandas as pd
import numpy as np

# Creating a synthetic version of the UCI Heart Disease Dataset based on report specs
np.random.seed(42)
n_samples = 303

data = {
    'age': np.random.randint(29, 78, n_samples),
    'sex': np.random.choice([0, 1], n_samples),
    'cp': np.random.choice([1, 2, 3, 4], n_samples),
    'trestbps': np.random.randint(94, 201, n_samples),
    'chol': np.random.randint(126, 565, n_samples),
    'fbs': np.random.choice([0, 1], n_samples),
    'restecg': np.random.choice([0, 1, 2], n_samples),
    'thalach': np.random.randint(71, 203, n_samples),
    'exang': np.random.choice([0, 1], n_samples),
    'oldpeak': np.random.uniform(0, 6.2, n_samples),
    'slope': np.random.choice([1, 2, 3], n_samples),
    'ca': np.random.choice([0, 1, 2, 3, np.nan], n_samples), # Including NaNs for preprocessing demo
    'thal': np.random.choice([1, 2, 3, np.nan], n_samples), # Including NaNs for preprocessing demo
    'target': np.random.choice([0, 1], n_samples)
}

df = pd.DataFrame(data)
df.to_csv('heart_disease.csv', index=False)
print("✅ dataset 'heart_disease.csv' created successfully!")