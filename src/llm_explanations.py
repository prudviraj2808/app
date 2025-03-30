from transformers import pipeline

# Initialize the text-generation pipeline using DistilGPT-2
generator = pipeline("text-generation", model="distilgpt2")

def generate_explanation(user_input, predictions):
    # Convert the user input DataFrame to a JSON-like string for clarity.
    input_str = user_input.to_json(orient="records", lines=True)
    prompt = (
        f"Patient data: {input_str}\n"
        f"Model Predictions: {predictions}\n"
        "Based on the above patient data and model predictions, provide a detailed explanation "
        "of the results, suggest potential coping mechanisms, and advise on next steps for mental health care."
    )
    # Generate text
    generated = generator(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
    return generated