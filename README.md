# Customer Churn Prediction with Explainable ML

A machine learning project that predicts customer churn probability and identifies customers who are at higher risk of leaving a telecom service.

## Project Overview

Customer churn prediction helps businesses identify customers who may leave their service and take proactive retention actions.

This project uses machine learning models to predict churn and explain the factors associated with churn risk.

## Features

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Categorical feature encoding
- Logistic Regression
- Random Forest
- XGBoost
- Model performance comparison
- SHAP-based model explainability
- High-risk customer identification
- Streamlit web application
- Churn probability prediction
- Low, Medium, and High risk classification
- Basic retention recommendations

## Dataset

The project uses the Telco Customer Churn dataset.

The dataset contains customer information such as:

- Tenure
- Contract type
- Internet service
- Monthly charges
- Total charges
- Payment method
- Customer demographics
- Churn status

## Machine Learning Models

The following models were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

### Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.38% | 64.76% | 57.49% | 60.91% | 83.57% |
| Random Forest | 76.83% | 55.45% | 65.24% | 59.95% | 82.09% |
| XGBoost | 79.46% | 63.08% | 54.81% | 58.66% | 83.84% |

Performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## Explainable AI

SHAP (SHapley Additive exPlanations) was used to understand which features have the strongest influence on the model's churn predictions.

The analysis helps identify factors associated with higher or lower predicted churn risk.

## High-Risk Customer Analysis

The project identifies customers with high predicted churn probability.

This can help businesses prioritize customers for targeted retention actions.

## Streamlit Application

The project includes a Streamlit web application where users can enter customer details and receive:

- Churn probability
- Churn risk level
- Basic retention recommendation

Risk levels are classified as:

- Low Churn Risk
- Medium Churn Risk
- High Churn Risk

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   └── check_data.py
│
├── app/
│   └── app.py
│
├── xgb_model.pkl
├── feature_columns.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app/app.py
```

The application will open in the browser.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Matplotlib
- Seaborn
- Joblib

## Project Outcome

The project demonstrates an end-to-end machine learning workflow covering data preprocessing, model training, evaluation, explainability, high-risk customer identification, and deployment through a Streamlit application.