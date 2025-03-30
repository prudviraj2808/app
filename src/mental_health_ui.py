import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from llm_explanations import generate_explanation

@st.cache_resource
def load_artifacts():
    """Load saved model, scaler, and selected features."""
    xgb_model = joblib.load(os.path.join("models/new_xgboost_model.pkl"))
    scaler = joblib.load(os.path.join("models/new_scaler.pkl"))
    selected_features = joblib.load(os.path.join("models/new_selected_features.pkl"))
    return xgb_model, scaler, selected_features

def predict_with_shap(user_input_df):
    """Make predictions and compute SHAP values for interpretation."""
    xgb_model, scaler, selected_features = load_artifacts()
    
    # Ensure the input DataFrame contains the expected features
    df_selected = user_input_df[selected_features]
    input_scaled = scaler.transform(df_selected)
    
    # Make predictions
    xgb_pred = xgb_model.predict(input_scaled)
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(input_scaled)
    
    return {
        "XGBoost Prediction": int(xgb_pred),
        "SHAP Values": shap_values,
        "Explainer": explainer,
        "Input Data": df_selected
    }

# --- Streamlit UI ---
st.title("Mental Health Prediction App")
st.markdown("Enter patient symptoms and details below:")

# Numeric input fields
phq_score = st.number_input("PHQ-9 Score", min_value=0, max_value=27, value=13)
gad_score = st.number_input("GAD-7 Score", min_value=0, max_value=21, value=10)
sofas = st.number_input("SOFAS Score", min_value=30, max_value=99, value=65)
duration_months = st.number_input("Duration (Months)", min_value=0.0, max_value=24.0, value=12.0)
past_episodes = st.number_input("Past Episodes", min_value=0, max_value=4, value=1)

# Categorical and boolean input fields
suicidality = st.selectbox("Suicidality", options=["Unknown", "Passive", "Active"])
treatment_failed = st.checkbox("Treatment Response Failed (2+ episodes)")
trauma = st.checkbox("Trauma")
substance_use = st.checkbox("Substance Use")

# Convert categorical values to numeric
suicidality_mapping = {"Unknown": 0, "Passive": 1, "Active": 2}

user_input = {
    "PHQ-9": phq_score,
    "GAD-7": gad_score,
    "SOFAS": sofas,
    "Duration_Months": duration_months,
    "Past_Episodes": past_episodes,
    "Suicidality": suicidality_mapping[suicidality],
    "Treatment_Response_Failed2+": int(treatment_failed),
    "Trauma": int(trauma),
    "Substance_Use": int(substance_use),
}

# Convert dictionary to DataFrame
user_input_df = pd.DataFrame(user_input, index=[0])

if st.button("Predict"):
    predictions = predict_with_shap(user_input_df)
    st.subheader("Model Predictions")
    st.write({"Prediction": predictions["XGBoost Prediction"]})

    # Generate and display the LLM explanation
    explanation = generate_explanation(user_input_df, predictions["XGBoost Prediction"])
    st.subheader("Explanation and Recommendations")
    st.write(explanation)

    # Display SHAP force plots for all classes and samples
    st.subheader("SHAP Force Plots for All Classes")
    explainer = predictions["Explainer"]
    shap_values = predictions["SHAP Values"]
    df_selected = predictions["Input Data"]

    # Ensure SHAP JS is loaded
    shap.initjs()

    # Loop through each sample and class
    for sample_index in range(shap_values.shape[0]):  # Loop over samples
        for class_index in range(shap_values.shape[2]):  # Loop over classes
            st.markdown(f"### Sample {sample_index}, Class {class_index}")
            
            force_plot = shap.force_plot(
                explainer.expected_value[class_index], 
                shap_values[sample_index, :, class_index], 
                df_selected.iloc[sample_index], 
                matplotlib=False  # Ensure it uses JS-based visualization
            )

            # Render the HTML in Streamlit using the .html() method
            shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
            components.html(shap_html, height=400)
