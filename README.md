# Credit-Default-SHAP-Project
Credit Default Prediction – Interpretable Machine Learning (SHAP Project)
This project predicts credit card default using XGBoost and RandomForest models, followed by global and local interpretability using SHAP (SHapley Additive exPlanations).

The project is structured according to explainable ML requirements for model transparency and fairness.

Models Used:
XGBoost Classifier
RandomForest Classifier
Evaluation Metrics
AUC Score
F1 Score
Confusion Matrix & Classification Report

Interpretability:
This project uses SHAP for:

1. Global Interpretability
Top 10 most important features
Summary plot
Feature importance bar plot
2.Local Interpretability
False Negative case
False Positive case
Borderline correct prediction

How to Run the Project:
Option 1 — Google Colab (Recommended)
Upload the dataset and notebook
Install packages using requirements.txt
Run cells from top to bottom
Option 2 — Local Machine
pip install -r requirements.txt
jupyter notebook project.ipynb

Key Insights:
Payment history (PAY_0, PAY_2, PAY_3) and recent payment amounts (PAY_AMT6, PAY_AMT4) are leading predictors.
SHAP reveals true feature influence more accurately than XGBoost built-in importance.
Local explanations highlight model errors and risk patterns.
