# Mental Health Prediction App

## Overview
The Mental Health Prediction App is a self-analysis tool designed to predict possible mental health conditions based on user-provided symptoms and details. The project combines robust machine learning techniques with a natural language explanation component. Two models—Random Forest and XGBoost—are developed for multi-class classification, while a pre-trained LLM (DistilGPT-2) generates detailed explanations and suggestions regarding the predictions.

This application is built for easy integration into chatbots or web-based interfaces, offering both command-line and interactive UI (Streamlit) options.

## Repository Structure
- **mental_health_ui.py**  
  Streamlit-based UI that allows users to enter patient data, view model predictions, and read generated explanations.

- **predict_mental_health.py**  
  Pre-processing datasets, training and testing the models with comparison as well as saving them.

- **llm_explanations.py**  
  Module containing the logic for generating natural language explanations using the Hugging Face Transformers pipeline with DistilGPT-2.

- **models/**  
  Directory containing:
  - Trained model artifacts (Random Forest and XGBoost).
  - Preprocessing artifacts such as the scaler and the list of selected features.
  - EDA plots (e.g., countplot, heatmap, SHAP summary plot).

- **data/raw**  
  Contains the raw dataset (e.g., “depression_anxiety_data.csv”) used for training and model development.

- **requirements.txt**  
  Lists all dependencies required to reproduce the project environment.

## Data Preparation & Modeling
### Data Preprocessing
- **Cleaning:** Duplicates are removed and missing values in numeric columns are filled with the median while categorical columns use the mode.
- **Encoding:** Categorical features are encoded using LabelEncoder.
- **EDA:** Exploratory Data Analysis is performed to visualize the distribution of target labels and feature correlations; plots are saved in the `models/` directory.
  
### Feature Engineering & Selection
- **Selection:** Unnecessary columns (e.g., id) are dropped. The top 10 features are selected based on feature importance obtained from a Random Forest classifier.
  
### Model Development
- **Models:** Two classifiers are built:
  - Random Forest Classifier (with hyperparameter tuning using GridSearchCV)
  - XGBoost Classifier (also tuned with GridSearchCV)
- **Evaluation:** Models are evaluated using Accuracy, Precision, Recall, F1 Score, and ROC-AUC metrics. SHAP analysis is performed to provide interpretability for the predictions.

### Old Pipeline: Target = 'suicidal' 
#### Achieving 94% Accuracy
- **Feature Selection:** Top 10 features are selected based on feature importance from a Random Forest classifier.
- **Data Splitting:** Data is split into training and testing sets.
- **Standardization:** Features are standardized using StandardScaler.
- **Hyperparameter Tuning:** GridSearchCV is used for hyperparameter tuning with 'accuracy' as the scoring metric.
- **Model Evaluation:** Models are evaluated using the `evaluate_model` function.

### New Pipeline: Target = 'anxiety_diagnosis' 
#### Achieving 98% Accuracy
- **Feature Selection:** Top 10 features are selected based on feature importance from a Random Forest classifier.
- **Data Splitting:** Data is split into training and testing sets.
- **Standardization:** Features are standardized using StandardScaler.
- **Class Imbalance Handling:** The ratio of negative to positive cases is computed for XGBoost to handle class imbalance.
- **Hyperparameter Tuning:** GridSearchCV is used for hyperparameter tuning with 'f1' as the scoring metric.
- **Model Evaluation:** Models are evaluated using the `evaluate_model` function.

### Model Comparison
- **Comparison:** Metrics from all four models (Old RF, Old XGB, New RF, New XGB) are combined into a single DataFrame for comparison.
- **Average Accuracy:** Average accuracy across all models is calculated and printed.
- **Visualization:** A grouped bar chart is plotted to compare the performance metrics of the models.

### SHAP Interpretation
- **SHAP Analysis:** SHAP analysis is performed for the new Random Forest model to provide interpretability for the predictions.
- **Visualization:** A SHAP summary plot is generated and saved.

### Saving Models and Artifacts
- **Saving:** The trained models, scaler, and selected features are saved using joblib.

## LLM Explanation Module
A natural language explanation is generated using a text-generation pipeline based on DistilGPT-2. The LLM component:
- Receives a prompt that summarizes the patient data and the model predictions.
- Outputs a detailed explanation that includes insights into the prediction, suggested coping mechanisms, and potential next steps for mental health care.

## Assignment Details
This project was developed as part of the Arogo AI AI/ML Engineer Intern Assignment with the following objectives:
- Build a self-analysis mental health model that predicts mental health conditions from user input.
- Compare and justify model performance (Random Forest vs. XGBoost).
- Integrate LLM-generated natural language explanations.
- Provide an interactive UI/CLI for user testing.

## Getting Started
### Requirements
- Python 3.7 or higher
### Install the required dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
### How to Run the Files
Run the prediction script:
```bash
python src/predict_mental_health.py
```
Launch the Streamlit UI:
```bash
streamlit run src/mental_health_ui.py
```