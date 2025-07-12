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

# Path to model weights (use relative path for Streamlit Cloud)
model_path = "."

# Safe loading with error handling
try:
    scaler = joblib.load(os.path.join(model_path, "scaler.pkl"))
    model = joblib.load(os.path.join(model_path, "voting_model.pkl"))
except Exception as e:
    st.error("❌ Failed to load model files. This may be due to Python version incompatibility or custom object references in the pickled files.")
    st.stop()

# Define feature inputs and their min/max ranges
feature_ranges = {
    "LLLH": (0.0, 1.0),
    "HTLH": (0.0, 1.0),
    "LERH": (0.0, 3.0),
    "LLRF": (0.0, 1.0),
    "LERF": (0.0, 3.0)
}

# Input form
with st.form("laminitis_form"):
    st.subheader("Enter Horse Diagnostic Features")
    inputs = {}
    for feature, (min_val, max_val) in feature_ranges.items():
        inputs[feature] = st.number_input(feature, min_value=min_val, max_value=max_val, step=0.1)
    submit = st.form_submit_button("Predict Risk")

# On submit
if submit:
    input_features = np.array([[inputs[f] for f in feature_ranges]])

    try:
        scaled_input = scaler.transform(input_features)
        prediction = model.predict(scaled_input)[0]
        proba = model.predict_proba(scaled_input)[0][1]  # Probability of positive class

        if prediction == 1:
            st.error(f"⚠️ High Risk of Laminitis (Confidence: {proba:.2%})")
        else:
            st.success(f"✅ Low Risk of Laminitis (Confidence: {1 - proba:.2%})")

        st.markdown("---")
        st.markdown("**Prediction Details:**")
        st.write(inputs)

    except ValueError as e:
        st.error(f"❌ Feature mismatch or transformation error: {str(e)}")
