# path: laminitis_app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Set up page config
st.set_page_config(
    page_title="Horse Laminitis Risk Prediction",
    layout="centered",
    initial_sidebar_state="auto"
)

# Title and description
st.title("🐎 Horse Laminitis Risk Prediction App")
st.markdown("""
This application uses a trained ensemble Voting Classifier to predict the risk of laminitis in horses.
Please enter the following features measured from the horse to get a prediction.
""")

# Path to model weights
#model_path = r"D:\\Medical_Imaging\\Risk Calculation\\Laminitis\\Weights"

# Load model components
# scaler = joblib.load(os.path.join(model_path, "scaler.pkl"))
# selector = joblib.load(os.path.join(model_path, "feature_selector.pkl"))
# model = joblib.load(os.path.join(model_path, "voting_model.pkl"))

import os
import joblib

joblib.dump(scaler, "scaler.pkl")
joblib.dump(selector, "feature_selector.pkl")
joblib.dump(voting_clf, "voting_model.pkl")

# Input form
with st.form("laminitis_form"):
    st.subheader("Enter Horse Diagnostic Features")
    LLLH = st.number_input("LLLH", min_value=0.0, max_value=1.0, step=0)
    HTLH = st.number_input("HTLH", min_value=0.0, max_value=1.0, step=0)
    LERH = st.number_input("LERH", min_value=0.0, max_value=3.0, step=0)
    LLRF = st.number_input("LLRF", min_value=0.0, max_value=1.0, step=0)
    LERF = st.number_input("LERF", min_value=0.0, max_value=3.0, step=0)
    submit = st.form_submit_button("Predict Risk")

# On submit
if submit:
    input_features = np.array([[LLLH, HTLH, LERH, LLRF, LERF]])
    scaled_input = scaler.transform(input_features)
    selected_input = selector.transform(scaled_input)
    prediction = model.predict(selected_input)[0]
    proba = model.predict_proba(selected_input)[0][1]  # Probability of positive class

    if prediction == 1:
        st.error(f"⚠️ High Risk of Laminitis (Confidence: {proba:.2%})")
    else:
        st.success(f"✅ Low Risk of Laminitis (Confidence: {1 - proba:.2%})")

    st.markdown("---")
    st.markdown("**Prediction Details:**")
    st.write({"LLLH": LLLH, "HTLH": HTLH, "LERH": LERH, "LLRF": LLRF, "LERF": LERF})
