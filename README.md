# IEEE-CIS Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## Overview

The **IEEE-CIS Fraud Detection System** is an end-to-end Machine Learning pipeline and interactive real-time dashboard developed as part of the **NTI Graduation Project**. The system covers the complete ML lifecycle — from raw data ingestion, cleaning, and feature engineering, through model training and evaluation, to a production-ready deployment consisting of a FastAPI backend and a Streamlit-powered interactive dashboard for real-time fraud prediction.

## Team 5 Members & Roles

| Member | Role | Responsibilities |
|---|---|---|
| **Mohamed Ahdy** | Data Engineering & EDA Lead | Data cleaning, missing values handling, exploratory data analysis (EDA), and feature engineering |
| **Abdelrahman Ramadan** | AI Modeling & Tuning Lead | Imbalanced data handling (SMOTE/Undersampling), baseline model training, and hyperparameter tuning |
| **Alfarouq Ibrahim** | Deployment & Product Lead | Joblib model serialization, FastAPI backend architecture, Streamlit UI/UX dashboard, and GitHub production pipeline |

## Dataset

This project uses the **IEEE-CIS Fraud Detection** dataset, provided by **Vesta Corporation**, containing real-world e-commerce transaction data labeled for fraudulent activity. The dataset combines transactional and identity information to support robust fraud detection modeling.

Source: [https://www.kaggle.com/competitions/ieee-fraud-detection](https://www.kaggle.com/competitions/ieee-fraud-detection)

## Project Structure

```
IEEE_Fraud_Detection/
├── models/
│   └── fraud_model.pkl
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Modeling.ipynb
│   └── 03_Deployment.ipynb
├── api.py
├── app.py
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/alfarouq637/IEEE_Fraud_Detection.git

# Navigate into the project directory
cd IEEE_Fraud_Detection

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

### Backend API

Start the FastAPI backend server:

```bash
python -m uvicorn api:app --reload
```

- API base URL: `http://localhost:8000`
- Interactive Swagger UI documentation: `http://localhost:8000/docs`

### Frontend UI

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

- Dashboard URL: `http://localhost:8501`

## Sample Test Data

Use the following sample feature array to test the prediction endpoint or dashboard input form:

```
0.5 0.1 0.9 0.3 0.8 0.2 0.4 0.7 0.6 0.1
```

## Methodology & Workflow

1. **Data Cleaning** — Handling missing values, correcting data types, and removing inconsistencies across the transactional and identity datasets.
2. **Feature Engineering** — Constructing informative features from raw transaction and identity attributes to improve model discriminative power.
3. **Imbalanced Data Handling** — Applying SMOTE (Synthetic Minority Over-sampling Technique) and Undersampling strategies to address the highly imbalanced fraud/non-fraud class distribution.
4. **Model Comparison** — Training and evaluating multiple algorithms, including Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost, to identify the best-performing model based on evaluation metrics.
5. **Real-Time Deployment** — Serializing the final model with Joblib and deploying it through a FastAPI backend, paired with a Streamlit dashboard for real-time, interactive fraud prediction.
