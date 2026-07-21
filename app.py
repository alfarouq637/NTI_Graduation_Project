import streamlit as st
import joblib
import numpy as np
import re

# configure page layout & security shield logo
st.set_page_config(
    page_title="Team 5 IEEE Fraud Detection",
    page_icon="https://cdn-icons-png.flaticon.com/512/2092/2092663.png",
    layout="wide"
)

# adaptive css for both light & dark modes
st.markdown('''
<style>
    .main-header {font-size: 38px; color: #FF4B4B; font-weight: bold;}
    .sub-header {font-size: 20px; opacity: 0.8;}
    .card {
        background: rgba(128, 128, 128, 0.1); 
        padding: 20px; 
        border-radius: 10px; 
        border-left: 5px solid #FF4B4B; 
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        min-height: 210px;
        dispplay: flex;
        flex-direction: column;
    }
    .team-name {font-size: 22px; font-weight: bold; color: #0083B8;}
    .role-text {font-size: 16px; font-weight: 600; opacity: 0.9;}
    .desc-text {font-size: 14px; opacity: 0.8; line-height: 1.5;}
</style>
''', unsafe_allow_html=True)

# sidebar navigation & team brand
st.sidebar.title("Team 5 Portal")
st.sidebar.markdown("### NTI IEEE Fraud Detection")
page = st.sidebar.radio("Navigation", ["Home & Team", "Fraud Detector", "Project Info"])

# load saved model
model = joblib.load("models/fraud_model.pkl")

# page 1 home & team members info
if page == "Home & Team":
    st.markdown('<p class="main-header">IEEE CIS Fraud Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Machine Learning System to Detect Financial Transaction Fraud</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">NTI Graduation Project</p>', unsafe_allow_html=True)
    st.write("---")

    st.header("Meet Team 5 Members")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card"><p class="team-name">Mohamed Ahdy</p><p class="role-text">Data Engineering & EDA Lead</p><p class="desc-text">Responsible for data cleaning preprocessing missing values & feature engineering</p></div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><p class="team-name">Abdelrahman Ramadan</p><p class="role-text">AI Modeling & Tuning Lead</p><p class="desc-text">Responsible for handling imbalanced data training baseline models & hyperparameter tuning</p></div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card"><p class="team-name">Alfarouq Ibrahim</p><p class="role-text">Deployment & Product Lead</p><p class="desc-text">Responsible for model serialization FastAPI APIs Streamlit portal & GitHub production</p></div>', unsafe_allow_html=True)

# page 2 real time fraud predictor
elif page == "Fraud Detector":
    st.markdown('<p class="main-header">Real-Time Transaction Analysis</p>', unsafe_allow_html=True)
    st.write("Enter financial transaction features below to simulate real-time AI detection")

    # input box for transaction features
    user_input = st.text_area("Transaction Features Input", placeholder="Example: 0.5 0.1 0.9 0.3 0.8 0.2 0.4 0.7 0.6 0.1", height=100)

    col1, col2 = st.columns([1, 4])
    with col1:
        predict_btn = st.button("Analyze Transaction", type="primary", use_container_width=True)

    if predict_btn:
        try:
            # extract clean floats from user string
            features = [float(val) for val in re.findall(r"[-+]?\d*\.\d+|\d+", user_input)]

            # validate features len & run model
            if len(features) > 0:
                data_array = np.array(features).reshape(1, -1)
                prediction = model.predict(data_array)[0]

                st.write("---")
                if prediction == 1:
                    st.error("ALERT: FRAUDULENT TRANSACTION DETECTED")
                else:
                    st.success("STATUS: NORMAL LEGITIMATE TRANSACTION")
            else:
                st.warning("Please input numeric features")
        except Exception:
            st.error("Error analyzing transaction data")

# page 3 general project specifications
elif page == "Project Info":
    st.markdown('<p class="main-header">Project Architecture</p>', unsafe_allow_html=True)

    st.info("Dataset: IEEE-CIS Fraud Detection from Kaggle")
    st.link_button("Dataset Link", "https://www.kaggle.com/competitions/ieee-fraud-detection?utm", type="primary")
    st.write("### Workflow Stages")
    st.write("1. **Data Understanding & Cleaning**: Removing outliers & handling missing identities")
    st.write("2. **Feature Engineering**: Creating transactional aggregations & encoding categoricals")
    st.write("3. **Model Training**: Comparing XGBoost LightGBM & Random Forest with SMOTE")
    st.write("4. **Realtime Deployment**: RESTful API via FastAPI & interactive UI via Streamlit")
