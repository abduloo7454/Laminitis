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

# Input form
with st.form("laminitis_form"):
    st.subheader("Enter Horse Diagnostic Features (Integer Only)")
    LLLH = st.number_input("LLLH", min_value=0, max_value=1, step=1, format="%d")
    HTLH = st.number_input("HTLH", min_value=0, max_value=1, step=1, format="%d")
    LERH = st.number_input("LERH", min_value=0, max_value=3, step=1, format="%d")
    LLRF = st.number_input("LLRF", min_value=0, max_value=1, step=1, format="%d")
    LERF = st.number_input("LERF", min_value=0, max_value=3, step=1, format="%d")
    submit = st.form_submit_button("Predict Risk")

# On submit
if submit:
    # Validate input types are integers
    values = [LLLH, HTLH, LERH, LLRF, LERF]
    if not all(isinstance(v, int) or v.is_integer() for v in values):
        st.warning("⚠️ Please enter only integer values for all features.")
        st.stop()

    input_features = np.array([[LLLH, HTLH, LERH, LLRF, LERF]])

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
        st.write({"LLLH": LLLH, "HTLH": HTLH, "LERH": LERH, "LLRF": LLRF, "LERF": LERF})

    except ValueError as e:
        st.error(f"❌ Feature mismatch or transformation error: {str(e)}")
