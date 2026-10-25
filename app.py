import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Sales Predictor App",
    page_icon="📈",
    layout="centered"
)

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load('model.sav')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading the model 'model.sav': {e}")
    model = None

# App layout and header
st.title("📈 Advertising Sales Predictor")
st.write("""
This app predicts **Sales** based on advertising budgets allocated to **TV**, **Radio**, and **Newspaper** media channels.
""")

st.divider()

# User Inputs
st.subheader("Enter Advertising Budgets ($ Thousand)")

col1, col2, col3 = st.columns(3)

with col1:
    tv = st.number_input("TV Budget", min_value=0.0, max_value=500.0, value=150.0, step=1.0)

with col2:
    radio = st.number_input("Radio Budget", min_value=0.0, max_value=500.0, value=25.0, step=1.0)

with col3:
    newspaper = st.number_input("Newspaper Budget", min_value=0.0, max_value=500.0, value=20.0, step=1.0)

# Prediction Action
st.divider()

if st.button("Predict Sales", type="primary"):
    if model is not None:
        # Prepare the input data as a DataFrame to keep feature names intact and avoid warnings
        input_data = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        st.success(f"### 📊 Estimated Sales: ${prediction:.2f}K")
    else:
        st.error("Model not found. Please ensure 'model.sav' is uploaded to the root directory.")
