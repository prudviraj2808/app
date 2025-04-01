import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
import numpy as np
import google.generativeai as genai

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
    df_selected = user_input_df[selected_features]
    input_scaled = scaler.transform(df_selected)
    xgb_pred_prob = xgb_model.predict_proba(input_scaled)
    predicted_class = np.argmax(xgb_pred_prob, axis=1)
    predicted_class_score = xgb_pred_prob[np.arange(len(xgb_pred_prob)), predicted_class]
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(input_scaled)
    
    return {
        "XGBoost Prediction Score": float(predicted_class_score[0]),
        "Predicted Class": int(predicted_class[0]),
        "SHAP Values": shap_values,
        "Explainer": explainer,
        "Input Data": df_selected
    }

st.title("Mental Health Prediction App")
st.markdown("Enter patient symptoms and details below:")

# Patient details
patient_name = st.text_input("Patient Name", key="patient_name")
patient_age = st.number_input("Patient Age", min_value=18, max_value=100, value=30, key="patient_age")
patient_gender = st.selectbox("Patient Gender", options=["Male", "Female", "Other"], key="patient_gender")
patient_id = st.text_input("Patient ID", key="patient_id")

# Symptom inputs
phq_score = st.number_input("PHQ-9 Score", min_value=0, max_value=27, value=13, key="phq_score")
gad_score = st.number_input("GAD-7 Score", min_value=0, max_value=21, value=10, key="gad_score")
sofas = st.number_input("SOFAS Score", min_value=30, max_value=99, value=65, key="sofas")
duration_months = st.number_input("Duration (Months)", min_value=0.0, max_value=24.0, value=12.0, key="duration_months")
past_episodes = st.number_input("Past Episodes", min_value=0, max_value=4, value=1, key="past_episodes")

# Categorical and boolean inputs
suicidality = st.selectbox("Suicidality", options=["Unknown", "Passive", "Active"], key="suicidality")
treatment_failed = st.checkbox("Treatment Response Failed (2+ episodes)", key="treatment_failed")
trauma = st.checkbox("Trauma", key="trauma")
substance_use = st.checkbox("Substance Use", key="substance_use")

# API Key input
api_key = st.text_input("Enter API Key for Explanation Generation", type="password", key="api_key")

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

user_input_df = pd.DataFrame(user_input, index=[0])

predictions = predict_with_shap(user_input_df)

# Create two columns for Predict and Reset buttons
col1, col2 = st.columns(2)

# When user clicks "Predict"
with col1:
    if st.button("Predict", key="predict_button"):
        # Store predictions and user inputs in session state to persist until reset
        st.session_state.predictions = predict_with_shap(user_input_df)
        st.session_state.user_inputs = {
            "Patient Name": patient_name,
            "Patient Age": patient_age,
            "Patient Gender": patient_gender,
            "Patient ID": patient_id,
            "PHQ-9 Score": phq_score,
            "GAD-7 Score": gad_score,
            "SOFAS Score": sofas,
            "Duration (Months)": duration_months,
            "Past Episodes": past_episodes,
            "Suicidality": suicidality,
            "Treatment Response Failed": treatment_failed,
            "Trauma": trauma,
            "Substance Use": substance_use
        }

# Add a reset button to clear session state and inputs
with col2:
    if st.button("Reset", key="reset_button"):
        # Clear session state
        st.session_state.clear()
        
        # Reset all input fields
        st.session_state["patient_name"] = ""
        st.session_state["patient_age"] = 30
        st.session_state["patient_gender"] = "Male"
        st.session_state["patient_id"] = ""
        st.session_state["phq_score"] = 13
        st.session_state["gad_score"] = 10
        st.session_state["sofas"] = 65
        st.session_state["duration_months"] = 12.0
        st.session_state["past_episodes"] = 1
        st.session_state["suicidality"] = "Unknown"
        st.session_state["treatment_failed"] = False
        st.session_state["trauma"] = False
        st.session_state["substance_use"] = False
        st.session_state["api_key"] = ""

# Display predictions and user inputs if they exist in session state
if "predictions" in st.session_state and "user_inputs" in st.session_state:
    st.subheader("User Inputs")
    st.write(st.session_state.user_inputs)

    st.subheader("Model Predictions")
    st.write({"Prediction Score": st.session_state.predictions["XGBoost Prediction Score"]})
    st.write({"Prediction Class": st.session_state.predictions["Predicted Class"]})

    explainer = st.session_state.predictions["Explainer"]
    shap_values = st.session_state.predictions["SHAP Values"]
    df_selected = st.session_state.predictions["Input Data"]

    class_index = st.session_state.predictions["Predicted Class"]
    fig, ax = plt.subplots(figsize=(8, 6))
    shap_values_class = shap_values[0][:, class_index]
    plt.barh(df_selected.columns, shap_values_class)
    plt.xlabel("SHAP Value")
    plt.ylabel("Feature")
    plt.title("SHAP Values for Predicted Class")
    plt.tight_layout()
    st.pyplot(fig)


def generate_explanation(user_input, prediction_score, predicted_class, api_key):
    """
    Generate an explanation using Google Gemini API based on patient data and predictions.
    
    Parameters:
        user_input (pd.DataFrame): DataFrame containing patient data.
        prediction_score (float): Predicted score from the ML model.
        predicted_class (str): Assigned stage/classification of mental health severity.
    
    Returns:
        str: AI-generated response with explanation and recommendations.
    """
    
    # Convert user input DataFrame to JSON-like string
    input_str = user_input.to_json(orient="records", lines=True)
    
    # Define the prompt
    prompt = (f"""
        You are a licensed mental health specialist with expertise in assessment, diagnosis, and treatment of various mental health conditions. 
        You have been assigned to work with a patient who has shared their personal data and medical history with you.
        
        You have access to the patient's comprehensive data {input_str} and predicted classification (Assigned_Stage) along with prediction scores.
        The data includes the following attributes with their definitions:
        
        • PHQ-9: Patient Health Questionnaire-9, a standardized questionnaire to assess the severity of depressive symptoms.
        • SOFAS: Social and Occupational Functioning Assessment Scale, a measure of an individual's level of social and occupational functioning.
        • Duration_Months: The length of time (in months) an individual has been experiencing mental health symptoms or episodes.
        • Suicidality: The presence and severity of suicidal thoughts, intentions, or behaviors.
        • Past_Episodes: The number of previous episodes or instances of mental health conditions.
        • Treatment_Response_Failed2+: Indicates whether an individual has failed to respond to two or more treatments for their mental health condition.
        • Trauma: Experiences of physical, emotional, or psychological trauma.
        • Substance_Use: The use or misuse of substances that can impact mental health.
        
        Based on this data, a machine learning model has generated a prediction score of {prediction_score} and classified the patient's mental health severity as {predicted_class} (Assigned_Stage).
        
        Your task is to:
        1. Provide a detailed explanation of the results, including the implications of the predicted score and classification.
        2. Suggest potential coping mechanisms and strategies tailored to the patient's specific needs and circumstances.
        3. Advise on next steps for mental health care, including recommendations for therapy, medication, or lifestyle changes.
        
        Please respond with a comprehensive and compassionate report that addresses the patient's unique situation and promotes their overall well-being.
    """)
    
    # Configure Gemini API
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text

def chat_with_model(user_input, prediction_score, predicted_class, api_key):
    """Interactive chat function for Streamlit UI."""
    model = genai.GenerativeModel("gemini-1.5-pro")

    # Only regenerate the explanation if the prediction has changed (avoid resetting chat history)
    if "last_prediction" not in st.session_state or st.session_state.last_prediction != (prediction_score, predicted_class):
        new_explanation = generate_explanation(user_input, prediction_score, predicted_class, api_key)
        
        # Update explanation history and reset the chat messages
        st.session_state.explanation_history = [new_explanation]
        st.session_state.messages = [{"role": "AI", "content": new_explanation}]
        
        # Update the last prediction to avoid redundant explanation regeneration
        st.session_state.last_prediction = (prediction_score, predicted_class)

    # Display explanation history
    st.subheader("🔍 Explanation History")
    for idx, explanation in enumerate(st.session_state.explanation_history):
        with st.expander(f"Explanation {idx + 1}"):
            st.markdown(explanation)

    # Display stored chat messages
    for message in st.session_state.messages:
        with st.chat_message("assistant" if message["role"] == "AI" else "user"):
            st.markdown(message["content"])

    # Handle user input and generate responses from the AI model
    user_query = st.chat_input("Ask me anything...")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        response = model.generate_content(user_query).text
        st.session_state.messages.append({"role": "AI", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if api_key:
    chat_with_model(user_input_df, predictions["XGBoost Prediction Score"], predictions["Predicted Class"], api_key)
