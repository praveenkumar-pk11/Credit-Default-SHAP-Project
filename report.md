# Interpretable Machine Learning – Credit Default Prediction (SHAP Analysis)

## 1. Project Overview
This project predicts **credit card default risk** using the UCI “Default of Credit Card Clients” dataset.  
Two machine learning models were trained:

- **XGBoost Classifier**
- **RandomForest Classifier**

The best model was interpreted using **SHAP (SHapley Additive exPlanations)** for both global and local interpretability.

---

## 2. Dataset Summary
- **Source:** UCI Machine Learning Repository  
- **Rows:** 30,000  
- **Columns:** 25  
- **Target Column:** `default` (1 = default, 0 = no default)

### Key Feature Groups
- Demographic details (AGE, SEX, EDUCATION)
- Credit Limit (LIMIT_BAL)
- Past Bill Amounts (BILL_AMT1–BILL_AMT6)
- Payment history delay (PAY_0–PAY_6)
- Past Payment Amounts (PAY_AMT1–PAY_AMT6)

---

## 3. Preprocessing
- Missing values handled using **median imputation**
- Only numeric features present (no categorical encoding needed)
- Scaling applied using **StandardScaler**
- Training/testing split: **75% / 25%**
- Dataset transformed into a numerical matrix for XGBoost/RandomForest

---

## 4. Model Evaluation

### ✔ XGBoost Results
- **AUC:** 0.7768  
- **F1 Score:** 0.4699  
- Strong performance for default detection  
- Better recall on positive default cases compared to RF

### ✔ RandomForest Results
- **AUC:** 0.7746  
- **F1 Score:** 0.4642  
- Very similar performance, but slightly lower than XGBoost

### **Selected Best Model: XGBoost**

---

## 5. Global SHAP Analysis

### Top 10 Most Influential Features
Identified from mean absolute SHAP values:

1. PAY_AMT6  
2. PAY_AMT4  
3. PAY_3  
4. PAY_2  
5. PAY_AMT3  
6. PAY_AMT1  
7. PAY_AMT2  
8. BILL_AMT1  
9. LIMIT_BAL  
10. PAY_0  

### Interpretation
- Larger **recent payment amounts** (PAY_AMT6, PAY_AMT4, PAY_AMT3, PAY_AMT1) reduce risk.
- Higher **payment delays** (PAY_0, PAY_2, PAY_3) increase risk of default.
- High **current bill amount** contributes to higher probability of default.
- **Higher credit limit** is associated with lower risk.

### Global SHAP Plots
- `plots/global_shap_summary.png`
- `plots/global_shap_bar.png`

---

## 6. Local SHAP Analysis

Three specific instances were analyzed:

### 6.1 False Negative (Actual = Default, Predicted = No Default)
Reasons for misclassification:
- Large recent payments outweighed indicators of debt
- Model perceived financial recovery, reducing predicted risk

### 6.2 False Positive (Actual = No Default, Predicted = Default)
Reasons:
- High outstanding bill plus multiple delayed payments
- Low recent payments strengthened default suspicion

### 6.3 Borderline Correct Prediction
- Probability close to 0.5  
- Balanced signals: moderate payments, moderate bills  
- No dominant feature → borderline prediction

### Local SHAP Plots
- `plots/local_false_negative.png`
- `plots/local_false_positive.png`
- `plots/local_borderline_correct.png`

---

## 7. Feature Importance Comparison

### XGBoost Gain Importance
- Focuses on how often features are used in tree splits
- Often overemphasizes features used early in the tree

### SHAP Importance
- Measures real contribution to each prediction  
- Consistent with model behavior across all samples  
- More reliable for interpretability and risk analysis

### Differences Observed
- XGBoost importance favored early split features (PAY_0, PAY_2)
- SHAP highlighted **payment amounts** as consistently important (PAY_AMT6, PAY_AMT4, etc.)

---

## 8. Business Recommendations

### Top 3 Risk Indicators (SHAP-Based)
1. Consecutive months with **low payment amounts**
2. Multiple **delayed payment cycles**
3. High outstanding **current bill amount**

### Recommended Business Strategy
**Implement an Early Risk Alert System**:  
Flag customers with:
- High bill amount  
- Low recent payments  
- At least 2 delayed payment cycles  

This enables proactive interventions like payment reminders or adjusted credit conditions.

---

## 9. Conclusion
This interpretable ML framework successfully identifies key drivers of credit default and provides actionable insights using SHAP.  
The project ensures transparency, model reliability, and compliance with explainability requirements.

---

