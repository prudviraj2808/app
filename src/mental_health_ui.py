import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
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
    
    # Ensure the input DataFrame contains the expected features
    df_selected = user_input_df[selected_features]
    input_scaled = scaler.transform(df_selected)
    
    # Make predictions
    xgb_pred = xgb_model.predict(input_scaled)
    
    xgb_pred_prob = xgb_model.predict_proba(input_scaled)  # Shape: (n_samples, n_classes)

    # Get predicted class index (class with highest probability)
    predicted_class = np.argmax(xgb_pred_prob, axis=1)  # Shape: (n_samples,)

    # Get probability score of the predicted class for each sample
    predicted_class_score = xgb_pred_prob[np.arange(len(xgb_pred_prob)), predicted_class]
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(input_scaled)
    
    return {
        "XGBoost Prediction": int(xgb_pred),
        "XGBoost Prediction Score": float(predicted_class_score[0]),
        "Predicted Class": int(predicted_class[0]),
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

# API Key input
api_key = st.text_input("Enter API Key for Explanation Generation", type="password")

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

predictions = predict_with_shap(user_input_df)  # Define globally

if st.button("Predict"):
    predictions = predict_with_shap(user_input_df)
    st.subheader("Model Predictions")
    st.write({"Prediction Score": predictions["XGBoost Prediction Score"]})
    st.write({"Prediction Class": predictions["Predicted Class"]})

    # Display SHAP waterfall plot for the first prediction
    explainer = predictions["Explainer"]
    shap_values = predictions["SHAP Values"]
    df_selected = predictions["Input Data"]

    class_index = predictions["XGBoost Prediction"]

    # Generate SHAP force plot
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.initjs()
    # Adjust the SHAP force plot to handle the correct index
    # Generate a bar plot for SHAP values of the predicted class
    shap_values_class = shap_values[0][:, class_index]
    feature_names = df_selected.columns

    # Create the bar plot
    plt.barh(feature_names, shap_values_class)
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

    # Check if it's a new prediction
    if "last_prediction" not in st.session_state or st.session_state.last_prediction != (prediction_score, predicted_class):
        # Clear explanation history and chat messages
        st.session_state.explanation_history = []
        st.session_state.messages = []

        # Generate new explanation
        new_explanation = generate_explanation(user_input, prediction_score, predicted_class, api_key)
        st.session_state.explanation_history.append(new_explanation)
        st.session_state.messages.append({"role": "AI", "content": new_explanation})

        # Update last prediction to avoid regenerating explanation unnecessarily
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

    # Handle user input
    user_query = st.chat_input("Ask me anything...")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})

        response = model.generate_content(user_query).text
        st.session_state.messages.append({"role": "AI", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)



if api_key:
    chat_with_model(user_input_df, predictions["XGBoost Prediction Score"], predictions["XGBoost Prediction"], api_key)
