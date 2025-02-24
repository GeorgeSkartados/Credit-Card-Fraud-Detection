#  Credit Card Fraud Detection using Machine Learning

## Project Overview
This project aims to detect fraudulent transactions in an e-commerce dataset using machine learning techniques. The goal is to develop a model that accurately classifies transactions as **fraudulent or non-fraudulent**, helping financial institutions minimize losses.

## Dataset
- **Source:** [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Imbalance:** Only **0.17%** of transactions are fraudulent.
- **Features:** 30 (including `Time`, `Amount`, and anonymized `V1`-`V28` features)
- **Target Variable:** `Class` (0 = Non-Fraud, 1 = Fraud)

## Data Preprocessing
- **Handling Class Imbalance:** Used **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic fraud samples.
- **Feature Scaling:** Standardized numerical features (`Amount` and `Time`).
- **Splitting:** 80% Training | 20% Testing.

## Machine Learning Models
We experimented with the following models:
-  **Logistic Regression**
-  **Decision Tree**
-  **Random Forest (Best Performing Model)**

## Model Performance Evaluation
To assess model performance, we used:
- **Confusion Matrix**
- **Precision-Recall Curve**
- **ROC Curve**
- **F1-Score (Main Metric)**

### **Best Model: Random Forest**
| Metric          | Score |
|----------------|-------|
| Accuracy       | 99.4% |
| Precision      | 92.8% |
| Recall         | 88.5% |
| F1-Score      | 90.6% |

##  Visualizations
- **Confusion Matrix:** Displays misclassifications.
- **ROC Curve:** Measures true positive vs. false positive rate.
- **Feature Importance:** Identifies key features influencing predictions.

##  How to Run the Project
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/GeorgeSkartados/credit-card-fraud-detection.git
