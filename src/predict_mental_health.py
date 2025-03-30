import os
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
import shap
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
import xgboost as xgb
xgb.set_config(verbosity=0)

###############################################
# 1. Common Data Loading and Preprocessing
###############################################
# Read the CSV file into a DataFrame.
data_path = os.path.join("data", "raw", "data.csv")
data = pd.read_csv(data_path)

# Encode all categorical variables using LabelEncoder.
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

###############################################
# 2. Define an Evaluation Function
###############################################
def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a model on test data and print overall performance metrics.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    try:
        roc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
    except Exception:
        roc = np.nan
    
    print(f"--- {model_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc}")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1, 'ROC-AUC': roc}

###############################################
# 3. NEW PIPELINE: Target = 'Assigned_Stage'
###############################################
X_new = data.drop(columns=['Assigned_Stage', 'id'], errors='ignore')
y_new = data['Assigned_Stage']

# Split the new data into training and testing sets.
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    X_new, y_new, test_size=0.2, random_state=42
)

# Standardize features for the new pipeline.
scaler_new = StandardScaler()
X_train_new = scaler_new.fit_transform(X_train_new)
X_test_new = scaler_new.transform(X_test_new)

# Define Hyperparameter Grid
param_grid = {
    'max_depth': [1, 2, 3, 5, 7, 8],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [10, 20, 30, 50, 100, 200],
    'subsample': [0.3, 0.4, 0.5],
    'colsample_bytree': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
}

# Define XGBoost Model
xgb = XGBClassifier(objective="multi:softmax", num_class=len(y_new.unique()), eval_metric="mlogloss", use_label_encoder=False)

grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train_new, y_train_new)
xgb_best_new = grid_search.best_estimator_

# Evaluate the new XGBoost model.
metrics_xgb_new = evaluate_model(xgb_best_new, X_test_new, y_test_new, "New XGBoost (Assigned_Stage)")

###############################################
# 4. SHAP Interpretation for XGBoost Model
###############################################
explainer = shap.TreeExplainer(xgb_best_new, X_train_new)
shap_values = explainer.shap_values(X_test_new, check_additivity=False)
shap.summary_plot(shap_values, X_test_new, show=False)
plt.savefig('models/shap_summary.png')
plt.close()

###############################################
# 5. Save New Model and Artifacts
###############################################
os.makedirs("models", exist_ok=True)
joblib.dump(xgb_best_new, os.path.join("models", "new_xgboost_model.pkl"))
joblib.dump(scaler_new, os.path.join("models", "new_scaler.pkl"))
joblib.dump(X_new.columns, os.path.join("models", "new_selected_features.pkl"))

print("Data preprocessing, model training, and evaluation complete. Model and artifacts saved.")
