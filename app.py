import streamlit as st
import joblib
import numpy as np
import os

# Set page configuration
st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")

st.title("Project 5: Breast Cancer Prediction System")
st.write("Enter the tumor features below to predict if it is Benign or Malignant.")

# 1. Load the Model with Error Handling (Feedback requirement)
model_path = 'model/breast_cancer_model.pkl'

@st.cache_resource # Caches the model to avoid reloading on every interaction
def load_model():
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please check directory structure.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

model = load_model()

# 2. Input Fields for the 5 Selected Features
# Using columns for a better layout
col1, col2 = st.columns(2)

with col1:
    radius_mean = st.number_input("Radius Mean", min_value=0.0, format="%.4f")
    perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, format="%.4f")
    area_mean = st.number_input("Area Mean", min_value=0.0, format="%.4f")

with col2:
    compactness_mean = st.number_input("Compactness Mean", min_value=0.0, format="%.4f")
    concavity_mean = st.number_input("Concavity Mean", min_value=0.0, format="%.4f")

# 3. Prediction Logic
if st.button("Predict Diagnosis"):
    if model is not None:
        try:
            # Create numpy array for prediction
            input_data = np.array([[radius_mean, perimeter_mean, area_mean, compactness_mean, concavity_mean]])
            
            # Predict (The pipeline handles the scaling automatically!)
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)

            # Display Result
            if prediction[0] == 1:
                st.error(f"Prediction: **Malignant** (Probability: {prediction_proba[0][1]:.2f})")
                st.write("The model suggests the tumor may be cancerous.")
            else:
                st.success(f"Prediction: **Benign** (Probability: {prediction_proba[0][0]:.2f})")
                st.write("The model suggests the tumor is likely non-cancerous.")
        
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Model is not loaded. Cannot predict.")

# Footer info
st.markdown("---")
st.caption("Educational purposes only. Do not use for medical diagnosis.")
