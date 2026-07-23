# IEEE-CIS Fraud Detection System

![License](https://img.shields.io/badge/License-MIT-yellow.svg)
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

## Live Demo

The dashboard is deployed and publicly accessible on Streamlit Community Cloud:

[**Team 5 IEEE Fraud Detection Streamlit**](https://ntigraduationproject.streamlit.app/)

## Dataset

This project uses the **IEEE-CIS Fraud Detection** dataset, provided by **Vesta Corporation**, containing real-world e-commerce transaction data labeled for fraudulent activity. The dataset combines transactional and identity information to support robust fraud detection modeling.

Source: [IEEE-Fraud-detection](https://www.kaggle.com/competitions/ieee-fraud-detection)

## Project Structure

```
IEEE_Fraud_Detection/
├── models/
│   └── fraud_model.pkl
├── notebooks/
│   ├── 01_Final Project.ipynb
│   └── 02_Deployment.ipynb
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

## Fraud Detector — Feature Input

The **Fraud Detector** page provides slider-based input for all **379 model features**, organized into collapsible sections:

| Section | Features | Description |
|---|---|---|
| Transaction Core | TransactionDT, TransactionAmt | Timestamp and dollar amount |
| Product Code | ProductCD (one-hot: H, R, S, W) | Product category |
| Card Information | card1–card5, card4/card6 (one-hot), addr1, addr2, dist1 | Payment card and billing details |
| Email Domains | P_emaildomain, R_emaildomain | Frequency-encoded email domains |
| Count Features | C1–C14 | Transaction count aggregations |
| Time Delta Features | D1–D5, D10, D11, D15 | Time gaps between events |
| Match Features | M1–M9, M4 (one-hot) | Name/address/email match flags |
| Vesta Features | V1–V137, V167–V321 | Anonymous Vesta payment features |
| Identity Features | id_01–id_38 (survivors) | Device and network identity |
| Device Features | DeviceType, DeviceInfo | Device type and info |
| Engineered: Time | TransactionHour, TransactionDay | Derived time features |
| Card Aggregation | card1_amt_mean/std, ratio | Per-card spending stats |
| UID Aggregation | uid_count/mean/std, ratio | Pseudo-customer aggregations |
| UID2 & Velocity | uid2_count/mean, ratio, time_since_last | Broader identity & velocity |

All fields have **default values** representing a typical legitimate transaction. You can expand any section, adjust individual features with sliders, and click **Analyze Transaction** to get a real-time prediction.

## Sample API Test

Send a POST request with 379 float features to the `/predict` endpoint:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5000000.0, 68.5, 0, 0, 0, 1, 10000, 321, 150, 0, 0, 1, 226, 0, 1, 0, 299, 87, 0, 100000, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 14, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 3, 100, 80, 1, 3, 100, 50, 1, 2, 100, 1, -1]}'
```

## Methodology & Workflow

1. **Data Cleaning** — Handling missing values, correcting data types, and removing inconsistencies across the transactional and identity datasets.
2. **Feature Engineering** — Constructing informative features from raw transaction and identity attributes to improve model discriminative power.
3. **Imbalanced Data Handling** — Applying SMOTE (Synthetic Minority Over-sampling Technique) and Undersampling strategies to address the highly imbalanced fraud/non-fraud class distribution.
4. **Model Comparison** — Training and evaluating multiple algorithms, including Logistic Regression and LightGBM , to identify the best-performing model based on evaluation metrics.
5. **Real-Time Deployment** — Serializing the final model with Joblib and deploying it through a FastAPI backend, paired with a Streamlit dashboard for real-time, interactive fraud prediction.
