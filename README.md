# Credit Card Fraud Detection Using Machine Learning

## ABSTRACT
Credit card fraud is a significant problem, with billions of dollars lost each year. Machine learning can be used to detect credit card fraud by identifying patterns indicative of fraudulent transactions. Credit card fraud refers to the physical loss of a credit card or the loss of sensitive credit card information. Various machine learning algorithms can be used for detection. This project aims to develop a machine learning model to detect fraudulent transactions. The model is trained on a dataset of historical credit card transactions and evaluated on a holdout dataset of unseen transactions.

**Keywords:** Credit Card Fraud Detection, Fraud Detection, Fraudulent Transactions, K-Nearest Neighbors, Support Vector Machine, Logistic Regression, Decision Tree.

---

## Overview
With the increasing use of credit cards in daily life, credit card companies must prioritize customer security. According to **Credit Card Statistics (2021)**, the number of global credit card users reached **2.8 billion** in 2019, with 70% of users owning a single card. Reports indicate that **credit card fraud in the U.S. increased by 44.7% in 2020**.

There are two main types of credit card fraud:
1. **New account fraud** – When an identity thief opens a credit card account under someone else’s name.
2. **Account takeover fraud** – When a thief gains access to an existing credit card account.

The alarming rise in fraud cases motivated this project to address the issue analytically using machine learning techniques to detect fraudulent transactions.

---

## Project Goals
The primary goal of this project is to accurately detect fraudulent credit card transactions, preventing customers from being charged for unauthorized purchases. Multiple machine learning techniques were applied to classify transactions as **fraudulent (1) or non-fraudulent (0)**. The performance of each model was compared using metrics such as **accuracy, precision, recall, and F1-score**.

Additionally, the project explores previous research on fraud detection and discusses different techniques used in identifying fraudulent transactions within large datasets.

---

## Data Source
- **Dataset Origin:** [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Transactions:** 284,808 rows over two days
- **Features:** 31 attributes (including PCA-transformed numerical variables for privacy protection)
- **Key Features:**
  - `Time` - Time elapsed between transactions
  - `Amount` - Transaction amount
  - `Class` - Target variable (0 = Non-Fraud, 1 = Fraud)

**Class Distribution:** Highly imbalanced, with fraudulent transactions accounting for only **0.17%** of the dataset.

---

## Algorithms Used
The following machine learning models were implemented and compared:
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression (LR)**
- **Support Vector Machine (SVM)**
- **Decision Tree (DT)**
- **Random Forest (Best Performing Model)**

---

## Model Performance Evaluation
To assess model effectiveness, we used:
- **Confusion Matrix**
- **Precision-Recall Curve**
- **ROC Curve**
- **F1-Score (Primary Metric)**

### **🏆 Best Model: Random Forest**
| Metric      | Score  |
|------------|--------|
| Accuracy   | 99.4%  |
| Precision  | 92.8%  |
| Recall     | 88.5%  |
| F1-Score   | 90.6%  |

---

## Future Work
Several enhancements can be made to improve this model:
- **Testing on larger datasets** with different characteristics
- **Optimizing hyperparameters** for better model performance
- **Integrating real-time fraud detection** using API deployment
- **Enhancing feature engineering** with additional data sources
- **Using deep learning models** (e.g., LSTMs) for detecting fraud in sequential data

One possible improvement is incorporating **telecom data** to analyze the cardholder’s location at the time of transactions. If a cardholder is detected in **Dubai** but their credit card is used in **New York**, the system could flag this as a potential fraud attempt.

---

## Conclusion
This project successfully developed a **machine learning-based fraud detection system**. The best-performing models, **Random Forest and Decision Tree**, achieved high accuracy in detecting fraudulent transactions. The results suggest that machine learning can effectively **enhance fraud detection systems**, minimizing financial losses and improving customer trust in banking security.

By implementing this system in real-world applications, financial institutions can significantly reduce fraudulent transactions and **ensure a safer digital payment ecosystem**.

---

## How to Run the Project
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/GeorgeSkartados/credit-card-fraud-detection.git
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Jupyter Notebook**
   ```bash
   jupyter notebook Fraud_Detection.ipynb
   ```

---

## Dependencies
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Imbalanced-learn (for SMOTE)
- Joblib (for model saving)

---

## Ethical Considerations
Fraud detection models must be fair and unbiased:
- **False Positives:** Can block legitimate transactions, frustrating customers.
- **False Negatives:** May allow fraud to go undetected, causing financial losses.
- **Bias in Training Data:** Must be minimized to ensure equal treatment of all transactions.

---

## License
This project is licensed under the **MIT License**.

---

## Author
[George Skartados](https://github.com/GeorgeSkartados)

