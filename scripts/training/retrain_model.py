"""
Laptop Price Predictor - Model Training Pipeline

Author: Sai Srikar Bhaskaruni
GitHub: https://github.com/SaiSrikar0/laptop-price-predictor
License: MIT

This module loads raw laptop data, performs feature engineering, trains a 
Lasso regression model, and saves the trained pipeline and reference data.

Usage:
    python scripts/training/retrain_model.py
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Lasso
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'laptop_data.csv'
ARTIFACTS_DIR = PROJECT_ROOT / 'data' / 'artifacts'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Load raw data
df = pd.read_csv(RAW_DATA_PATH)

# Drop unnamed index column if present
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Basic cleaning
# Ram and Weight
if df['Ram'].dtype == object:
    df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype(int)
if df['Weight'].dtype == object:
    df['Weight'] = df['Weight'].str.replace('kg', '', regex=False).astype(float)

# Touchscreen and Ips from ScreenResolution column
if 'ScreenResolution' in df.columns:
    df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in str(x) else 0)
    df['Ips'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in str(x) else 0)

    # Extract X_res and Y_res
    new = df['ScreenResolution'].str.split('x', n=1, expand=True)
    df['X_res'] = new[0]
    df['Y_res'] = new[1]

    # Clean X_res
    df['X_res'] = df['X_res'].astype(str).str.replace(',', '').str.findall(r'(\d+\.?\d+)').apply(lambda x: x[0] if len(x) > 0 else '0')
    df['X_res'] = df['X_res'].astype(int)
    df['Y_res'] = df['Y_res'].astype(str).str.findall(r'(\d+\.?\d+)').apply(lambda x: x[0] if len(x) > 0 else '0')
    df['Y_res'] = df['Y_res'].astype(int)

    # ppi
    df['ppi'] = (((df['X_res'] ** 2) + (df['Y_res'] ** 2)) ** 0.5) / df['Inches']
else:
    # fallback if ScreenResolution missing
    df['Touchscreen'] = 0
    df['Ips'] = 0
    df['ppi'] = df['Inches']

# CPU brand
# Create small cpu name and brand like the notebook
if 'Cpu' in df.columns:
    df['Cpu Name'] = df['Cpu'].apply(lambda x: " ".join(str(x).split()[0:3]))

    def fetch_processor(text):
        text = str(text)
        if text in ('Intel Core i7', 'Intel Core i5', 'Intel Core i3'):
            return text
        else:
            if text.split()[0] == 'Intel':
                return 'Other Intel Processor'
            else:
                return 'AMD Processor'

    df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)
else:
    df['Cpu brand'] = 'Other'

# Memory -> HDD and SSD
if 'Memory' in df.columns:
    df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
    df['Memory'] = df['Memory'].str.replace('GB', '', regex=False)
    df['Memory'] = df['Memory'].str.replace('TB', '000', regex=False)
    new_mem = df['Memory'].str.split('+', n=1, expand=True)
    df['first'] = new_mem[0].str.strip()
    df['second'] = new_mem[1].fillna('0')

    # flags
    df['Layer1HDD'] = df['first'].apply(lambda x: 1 if 'HDD' in str(x) else 0)
    df['Layer1SSD'] = df['first'].apply(lambda x: 1 if 'SSD' in str(x) else 0)
    df['Layer1Hybrid'] = df['first'].apply(lambda x: 1 if 'Hybrid' in str(x) else 0)
    df['Layer1Flash_Storage'] = df['first'].apply(lambda x: 1 if 'Flash Storage' in str(x) else 0)

    df['first'] = df['first'].str.replace(r'\D', '', regex=True)
    df['second'] = df['second'].str.replace(r'\D', '', regex=True)
    df['first'] = df['first'].replace('', '0')
    df['second'] = df['second'].replace('', '0')
    df['first'] = df['first'].astype(int)
    df['second'] = df['second'].astype(int)

    df['HDD'] = (df['first'] * df['Layer1HDD'] + df['second'] * df['second'].apply(lambda x: 0))
    df['SSD'] = (df['first'] * df['Layer1SSD'] + df['second'] * df['second'].apply(lambda x: 0))

    # The above is conservative: if first contains SSD or HDD we count it; we won't try to parse both layers perfectly here.
else:
    df['HDD'] = 0
    df['SSD'] = 0

# Gpu brand
if 'Gpu' in df.columns:
    df['Gpu brand'] = df['Gpu'].apply(lambda x: str(x).split()[0])
    df = df[df['Gpu brand'] != 'ARM']
else:
    df['Gpu brand'] = 'Other'

# OS mapping
if 'OpSys' in df.columns:
    def cat_os(inp):
        if inp in ('Windows 10', 'Windows 7', 'Windows 10 S'):
            return 'Windows'
        elif inp in ('macOS', 'Mac OS X'):
            return 'Mac'
        else:
            return 'Others/No OS/Linux'
    df['os'] = df['OpSys'].apply(cat_os)
else:
    df['os'] = 'Others/No OS/Linux'

# Drop intermediate columns that are not needed
for col in ['Cpu', 'Cpu Name', 'ScreenResolution', 'X_res', 'Y_res', 'first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid', 'Layer1Flash_Storage', 'OpSys', 'Memory']:
    if col in df.columns:
        try:
            df.drop(columns=[col], inplace=True)
        except Exception:
            pass

# Ensure Price exists
if 'Price' not in df.columns:
    raise ValueError('Price column not found in laptop_data.csv')

# Prepare X and y similar to notebook
y = np.log(df['Price'])

# Drop any object dtypes not explicitly handled in categorical columns
possible_obj_cols = df.select_dtypes(include=['object']).columns.tolist()

# We'll keep only the expected categorical columns; drop other object columns
expected_cat = ['Company', 'TypeName', 'Cpu brand', 'Gpu brand', 'os']
for col in possible_obj_cols:
    if col not in expected_cat and col != 'Price':
        try:
            df.drop(columns=[col], inplace=True)
        except Exception:
            pass

X = df.drop(columns=['Price'])
y = np.log(df['Price'])

# Define categorical and numeric columns for transformer by name
cat_cols = ['Company', 'TypeName', 'Cpu brand', 'Gpu brand', 'os']
num_cols = [c for c in X.columns if c not in cat_cols]

# ColumnTransformer
ct = ColumnTransformer(transformers=[
    ('ohe', OneHotEncoder(sparse_output=False, drop='first'), cat_cols)
], remainder='passthrough')

model = Lasso(alpha=0.001)
pipe = Pipeline([('pre', ct), ('model', model)])

# Fit
pipe.fit(X, y)

# Save df and pipe
pickle.dump(df, open(ARTIFACTS_DIR / 'df.pkl', 'wb'))
pickle.dump(pipe, open(ARTIFACTS_DIR / 'pipe.pkl', 'wb'))
print('Saved df.pkl and pipe.pkl in data/artifacts')
