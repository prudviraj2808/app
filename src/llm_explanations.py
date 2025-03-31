import google.generativeai as genai
import pandas as pd

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
    
    # Initialize the model
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    
    return response.text
