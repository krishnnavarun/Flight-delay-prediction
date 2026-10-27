"""
Generates the comprehensive Tamil Nadu & Indian Domestic Flight Delay Analysis Jupyter Notebook.
"""

import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Title & Abstract
cells.append(nbf.v4.new_markdown_cell("""# ✈️ AeroPredict: Indian Domestic Flight Delay Prediction & Analytics
### Focused Case Study: Tamil Nadu Aviation Network (Coimbatore, Chennai, Madurai, Tiruchirappalli)
**Author / Project Team**: Machine Learning for Civil Aviation Research  
**Benchmark Standard**: Directorate General of Civil Aviation (DGCA) 15-Minute Threshold Standard

---

## 📌 Executive Summary
Flight delays impose massive operational costs on commercial airlines and severe disruption on passengers. This notebook provides an end-to-end Data Science and Machine Learning workflow analyzing domestic flights across **Tamil Nadu** (Coimbatore CJB, Chennai MAA, Madurai IXM, Tiruchirappalli TRZ, Tuticorin TCR, Salem SXV) and connected metropolitan hubs (Bengaluru BLR, Mumbai BOM, Delhi DEL, Hyderabad HYD).
"""))

# Cell 1: Environment & Imports
cells.append(nbf.v4.new_code_cell("""# 1. Imports and Setup
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
%matplotlib inline
print("Libraries imported successfully!")
"""))

# Cell 2: Load Dataset
cells.append(nbf.v4.new_markdown_cell("""## 2. Dataset Ingestion & Overview"""))
cells.append(nbf.v4.new_code_cell("""data_path = 'Data/indian_flights_data.csv'
df = pd.read_csv(data_path)
print(f"Total Flight Records: {len(df):,}")
df.head(5)
"""))

# Cell 3: Metrics summary
cells.append(nbf.v4.new_code_cell("""print(f"Overall On-Time Performance (OTP): {(1 - df['delayed'].mean())*100:.2f}%")
print(f"Overall Delay Rate (>15 mins): {df['delayed'].mean()*100:.2f}%")
"""))

nb['cells'] = cells

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Tamil_Nadu_Flight_Delay_Analysis.ipynb')
with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Successfully generated Jupyter Notebook at: {out_path}")

