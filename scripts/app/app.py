"""
Laptop Price Predictor - Streamlit Web Application

Author: Sai Srikar Bhaskaruni
GitHub: https://github.com/SaiSrikar0/laptop-price-predictor
License: MIT

This module implements the interactive web interface for predicting laptop prices
based on hardware specifications using a trained Lasso regression model.
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# import the model
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / 'data' / 'artifacts'

pipe = pickle.load(open(ARTIFACTS_DIR / 'pipe.pkl', 'rb'))
df = pickle.load(open(ARTIFACTS_DIR / 'df.pkl', 'rb'))

st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type_name = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.slider('Screensize in inches', 10.0, 18.0, 13.0)
# Inches feature expected by the pipeline
inches = float(screen_size)

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',df['Cpu brand'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['Gpu brand'].unique())

os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    
    # Create DataFrame with proper column names
    query = pd.DataFrame([[company, type_name, inches, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]], 
                         columns=['Company', 'TypeName', 'Inches', 'Ram', 'Weight', 'Touchscreen', 'Ips', 'ppi', 'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os'])

    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))

