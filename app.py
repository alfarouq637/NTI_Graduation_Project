import streamlit as st
import joblib
import numpy as np

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
        display: flex;
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


# load saved model with caching and error handling
@st.cache_resource
def load_model():
    try:
        return joblib.load("models/fraud_model.pkl")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


model = load_model()

# ─────────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS — matches the 379 features from Notebook 01
# ─────────────────────────────────────────────────────────────────

# The ordered list of all 379 feature names expected by the model.
# This order MUST match the training data column order.
FEATURE_NAMES = [
    # --- Transaction Core (idx 0-1) ---
    "TransactionDT", "TransactionAmt",
    # --- Product Code one-hot (idx 2-5) ---
    "ProductCD_H", "ProductCD_R", "ProductCD_S", "ProductCD_W",
    # --- Card Info (idx 6-18) ---
    "card1", "card2", "card3",
    "card4_discover", "card4_mastercard", "card4_visa",
    "card5",
    "card6_credit", "card6_debit", "card6_debit or credit",
    "addr1", "addr2", "dist1",
    # --- Email Domains freq-encoded (idx 19-20) ---
    "P_emaildomain", "R_emaildomain",
    # --- Count Features C1-C14 (idx 21-34) ---
    *[f"C{i}" for i in range(1, 15)],
    # --- Time Delta Features D (idx 35-42) ---
    "D1", "D2", "D3", "D4", "D5", "D10", "D11", "D15",
    # --- Match Features M (idx 43-52) ---
    "M1", "M2", "M3", "M4_M1", "M4_M2", "M5", "M6", "M7", "M8", "M9",
    # --- Vesta Features V1-V137 (idx 53-189) ---
    *[f"V{i}" for i in range(1, 138)],
    # --- Vesta Features V167-V321 (idx 190-344) ---
    *[f"V{i}" for i in range(167, 322)],
    # --- Identity Features (idx 345-363) ---
    "id_01", "id_02", "id_05", "id_06", "id_11",
    "id_12", "id_13", "id_15", "id_16", "id_17", "id_19", "id_20",
    "id_28", "id_29", "id_31", "id_35", "id_36", "id_37", "id_38",
    # --- Device Features (idx 364-365) ---
    "DeviceType_mobile", "DeviceInfo",
    # --- Engineered: Time (idx 366-367) ---
    "TransactionHour", "TransactionDay",
    # --- Engineered: Card Aggregation (idx 368-370) ---
    "card1_amt_mean", "card1_amt_std", "TransactionAmt_to_mean_card1",
    # --- Engineered: UID Aggregation (idx 371-374) ---
    "uid_count", "uid_amt_mean", "uid_amt_std", "TransactionAmt_to_mean_uid",
    # --- Engineered: UID2 & Velocity (idx 375-378) ---
    "uid2_count", "uid2_amt_mean", "TransactionAmt_to_mean_uid2",
    "time_since_last_uid_txn",
]

assert len(FEATURE_NAMES) == 379, f"Expected 379 features, got {len(FEATURE_NAMES)}"

# Default values — representative of a typical legitimate transaction
# (medians/modes derived from the IEEE-CIS training data)
DEFAULTS = {
    # Transaction Core
    "TransactionDT": 5000000.0,
    "TransactionAmt": 68.5,
    # Product Code one-hot (W is most common — so its dummy = 1, rest = 0)
    "ProductCD_H": 0, "ProductCD_R": 0, "ProductCD_S": 0, "ProductCD_W": 1,
    # Card Info
    "card1": 10000.0, "card2": 321.0, "card3": 150.0,
    "card4_discover": 0, "card4_mastercard": 0, "card4_visa": 1,
    "card5": 226.0,
    "card6_credit": 0, "card6_debit": 1, "card6_debit or credit": 0,
    "addr1": 299.0, "addr2": 87.0, "dist1": 0.0,
    # Email Domains (frequency encoded — median frequency)
    "P_emaildomain": 100000.0, "R_emaildomain": 0.0,
    # Count Features (median ~1)
    **{f"C{i}": 1.0 for i in range(1, 15)},
    # Time Delta Features
    "D1": 14.0, "D2": 14.0, "D3": 0.0, "D4": 0.0, "D5": 0.0,
    "D10": 0.0, "D11": 0.0, "D15": 0.0,
    # Match Features (binary — default 0)
    "M1": 0, "M2": 0, "M3": 0, "M4_M1": 0, "M4_M2": 0,
    "M5": 0, "M6": 0, "M7": 0, "M8": 0, "M9": 0,
    # Vesta features (default 0.0)
    **{f"V{i}": 0.0 for i in range(1, 138)},
    **{f"V{i}": 0.0 for i in range(167, 322)},
    # Identity Features
    "id_01": 0.0, "id_02": 0.0, "id_05": 0.0, "id_06": 0.0, "id_11": 0.0,
    "id_12": 0, "id_13": 0.0, "id_15": 0, "id_16": 0,
    "id_17": 0.0, "id_19": 0.0, "id_20": 0.0,
    "id_28": 0, "id_29": 0, "id_31": 0.0, "id_35": 0, "id_36": 0, "id_37": 0, "id_38": 0,
    # Device Features
    "DeviceType_mobile": 0, "DeviceInfo": 0.0,
    # Engineered: Time
    "TransactionHour": 12, "TransactionDay": 3,
    # Engineered: Card Aggregation
    "card1_amt_mean": 100.0, "card1_amt_std": 80.0, "TransactionAmt_to_mean_card1": 1.0,
    # Engineered: UID Aggregation
    "uid_count": 3.0, "uid_amt_mean": 100.0, "uid_amt_std": 50.0, "TransactionAmt_to_mean_uid": 1.0,
    # Engineered: UID2 & Velocity
    "uid2_count": 2.0, "uid2_amt_mean": 100.0, "TransactionAmt_to_mean_uid2": 1.0,
    "time_since_last_uid_txn": -1.0,
}

# Fraud detection threshold from Notebook 01 Section 21 (max F1)
FRAUD_THRESHOLD = 0.77

# ─────────────────────────────────────────────────────────────────
# HELPER — create a slider/toggle and store value in session state
# ─────────────────────────────────────────────────────────────────


def _binary_toggle(label, key, default=0):
    """Render a binary 0/1 select box."""
    return st.selectbox(label, options=[0, 1], index=default, key=key)


def _float_slider(label, key, default=0.0, min_val=-1.0, max_val=1000.0, step=0.1):
    """Render a float number input."""
    return st.number_input(label, min_value=min_val, max_value=max_val,
                           value=float(default), step=step, key=key)


def _int_slider(label, key, default=0, min_val=0, max_val=100000):
    """Render an integer number input."""
    return st.number_input(label, min_value=min_val, max_value=max_val,
                           value=int(default), step=1, key=key)


# ─────────────────────────────────────────────────────────────────
# PAGE 1 — Home & Team
# ─────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────
# PAGE 2 — Fraud Detector (slider-based input)
# ─────────────────────────────────────────────────────────────────
elif page == "Fraud Detector":
    st.markdown('<p class="main-header">Real-Time Transaction Analysis</p>', unsafe_allow_html=True)
    st.write("Adjust transaction features below using the expandable sections, then click **Analyze Transaction**.")
    st.write("All fields have default values representing a typical legitimate transaction.")

    # Collect feature values from UI
    values = dict(DEFAULTS)  # start with defaults

    # ── Section 1: Transaction Core ──
    with st.expander("📋 Transaction Core", expanded=False):
        st.caption("Basic transaction timestamp and amount.")
        c1, c2 = st.columns(2)
        with c1:
            values["TransactionDT"] = _float_slider("TransactionDT (seconds since reference)", "TransactionDT",
                                                     default=5000000.0, min_val=0.0, max_val=20000000.0, step=1000.0)
        with c2:
            values["TransactionAmt"] = _float_slider("TransactionAmt ($)", "TransactionAmt",
                                                      default=68.5, min_val=0.0, max_val=50000.0, step=0.5)

    # ── Section 2: Product Code ──
    with st.expander("🏷️ Product Code (One-Hot Encoded)", expanded=False):
        st.caption("Select one product code. W=most common. One-hot encoded with drop_first (C is baseline).")
        pc = st.radio("Product Code", ["C (baseline)", "H", "R", "S", "W"], index=4,
                       key="product_code_radio", horizontal=True)
        values["ProductCD_H"] = 1 if pc == "H" else 0
        values["ProductCD_R"] = 1 if pc == "R" else 0
        values["ProductCD_S"] = 1 if pc == "S" else 0
        values["ProductCD_W"] = 1 if pc == "W" else 0

    # ── Section 3: Card Information ──
    with st.expander("💳 Card Information", expanded=False):
        st.caption("Payment card details, billing address, and distance.")
        c1, c2, c3 = st.columns(3)
        with c1:
            values["card1"] = _float_slider("card1", "card1", 10000.0, 0.0, 20000.0, 1.0)
            values["card2"] = _float_slider("card2", "card2", 321.0, 0.0, 600.0, 1.0)
            values["card3"] = _float_slider("card3", "card3", 150.0, 0.0, 300.0, 1.0)
            values["card5"] = _float_slider("card5", "card5", 226.0, 0.0, 500.0, 1.0)
        with c2:
            card4 = st.radio("card4 (Card Network)", ["american express (baseline)", "discover", "mastercard", "visa"],
                              index=3, key="card4_radio")
            values["card4_discover"] = 1 if card4 == "discover" else 0
            values["card4_mastercard"] = 1 if card4 == "mastercard" else 0
            values["card4_visa"] = 1 if card4 == "visa" else 0

            card6 = st.radio("card6 (Card Type)", ["charge card (baseline)", "credit", "debit", "debit or credit"],
                              index=2, key="card6_radio")
            values["card6_credit"] = 1 if card6 == "credit" else 0
            values["card6_debit"] = 1 if card6 == "debit" else 0
            values["card6_debit or credit"] = 1 if card6 == "debit or credit" else 0
        with c3:
            values["addr1"] = _float_slider("addr1 (billing region)", "addr1", 299.0, 0.0, 500.0, 1.0)
            values["addr2"] = _float_slider("addr2 (billing country)", "addr2", 87.0, 0.0, 102.0, 1.0)
            values["dist1"] = _float_slider("dist1 (distance)", "dist1", 0.0, 0.0, 10000.0, 1.0)

    # ── Section 4: Email Domains (frequency encoded) ──
    with st.expander("📧 Email Domains (Frequency Encoded)", expanded=False):
        st.caption("Frequency-encoded email domain counts from training data.")
        c1, c2 = st.columns(2)
        with c1:
            values["P_emaildomain"] = _float_slider("P_emaildomain (purchaser)", "P_emaildomain",
                                                      100000.0, 0.0, 300000.0, 100.0)
        with c2:
            values["R_emaildomain"] = _float_slider("R_emaildomain (recipient)", "R_emaildomain",
                                                      0.0, 0.0, 300000.0, 100.0)

    # ── Section 5: Count Features (C1-C14) ──
    with st.expander("🔢 Count Features (C1–C14)", expanded=False):
        st.caption("Transaction count features measuring various grouping aggregations.")
        cols = st.columns(4)
        for idx, i in enumerate(range(1, 15)):
            with cols[idx % 4]:
                values[f"C{i}"] = _float_slider(f"C{i}", f"C{i}", 1.0, 0.0, 5000.0, 1.0)

    # ── Section 6: Time Delta Features (D) ──
    with st.expander("⏱️ Time Delta Features (D)", expanded=False):
        st.caption("Time deltas between various transaction events.")
        d_features = ["D1", "D2", "D3", "D4", "D5", "D10", "D11", "D15"]
        d_defaults = {"D1": 14.0, "D2": 14.0, "D3": 0.0, "D4": 0.0, "D5": 0.0,
                      "D10": 0.0, "D11": 0.0, "D15": 0.0}
        cols = st.columns(4)
        for idx, d in enumerate(d_features):
            with cols[idx % 4]:
                values[d] = _float_slider(d, d, d_defaults[d], -1.0, 1000.0, 1.0)

    # ── Section 7: Match Features (M) ──
    with st.expander("🔐 Match Features (M1–M9)", expanded=False):
        st.caption("Binary match indicators for names, addresses, and email. M4 is one-hot encoded (M0=baseline).")
        cols = st.columns(5)
        binary_m = ["M1", "M2", "M3", "M5", "M6", "M7", "M8", "M9"]
        for idx, m in enumerate(binary_m):
            with cols[idx % 5]:
                values[m] = _binary_toggle(f"{m} (T=1 / F=0)", f"m_{m}")
        # M4 one-hot
        with cols[3]:
            m4 = st.radio("M4", ["M0 (baseline)", "M1", "M2"], index=0, key="m4_radio")
            values["M4_M1"] = 1 if m4 == "M1" else 0
            values["M4_M2"] = 1 if m4 == "M2" else 0

    # ── Section 8: Vesta Features V1-V137 ──
    with st.expander("🔬 Vesta Features V1–V137 (137 features)", expanded=False):
        st.caption("Anonymous Vesta payment features. Default is 0.0 for all.")
        cols = st.columns(5)
        for idx, i in enumerate(range(1, 138)):
            with cols[idx % 5]:
                values[f"V{i}"] = _float_slider(f"V{i}", f"V{i}", 0.0, -5.0, 5000.0, 0.1)

    # ── Section 9: Vesta Features V167-V321 ──
    with st.expander("🔬 Vesta Features V167–V321 (155 features)", expanded=False):
        st.caption("Anonymous Vesta payment features (second block). Default is 0.0 for all.")
        cols = st.columns(5)
        for idx, i in enumerate(range(167, 322)):
            with cols[idx % 5]:
                values[f"V{i}"] = _float_slider(f"V{i}", f"V{i}", 0.0, -5.0, 5000.0, 0.1)

    # ── Section 10: Identity Features ──
    with st.expander("🆔 Identity Features", expanded=False):
        st.caption("Device and network identity information.")
        # Numeric IDs
        numeric_ids = ["id_01", "id_02", "id_05", "id_06", "id_11", "id_13", "id_17", "id_19", "id_20"]
        cols = st.columns(4)
        for idx, fid in enumerate(numeric_ids):
            with cols[idx % 4]:
                values[fid] = _float_slider(fid, fid, 0.0, -100.0, 100000.0, 1.0)
        st.write("---")
        st.caption("Binary identity features (Found=1 / NotFound=0):")
        binary_ids = ["id_12", "id_15", "id_16", "id_28", "id_29", "id_35", "id_36", "id_37", "id_38"]
        cols = st.columns(5)
        for idx, fid in enumerate(binary_ids):
            with cols[idx % 5]:
                values[fid] = _binary_toggle(fid, f"id_{fid}")
        st.write("---")
        st.caption("Label-encoded identity feature:")
        values["id_31"] = _float_slider("id_31 (browser/device label encoded)", "id_31", 0.0, 0.0, 200.0, 1.0)

    # ── Section 11: Device Features ──
    with st.expander("📱 Device Features", expanded=False):
        st.caption("Device type and info (frequency encoded).")
        c1, c2 = st.columns(2)
        with c1:
            values["DeviceType_mobile"] = _binary_toggle("DeviceType (0=desktop / 1=mobile)", "DeviceType_mobile")
        with c2:
            values["DeviceInfo"] = _float_slider("DeviceInfo (frequency encoded)", "DeviceInfo",
                                                  0.0, 0.0, 100000.0, 1.0)

    # ── Section 12: Engineered — Time Features ──
    with st.expander("🕐 Engineered: Time Features", expanded=False):
        st.caption("Derived from TransactionDT.")
        c1, c2 = st.columns(2)
        with c1:
            values["TransactionHour"] = _int_slider("TransactionHour (0–23)", "TransactionHour", 12, 0, 23)
        with c2:
            values["TransactionDay"] = _int_slider("TransactionDay (0=Mon – 6=Sun)", "TransactionDay", 3, 0, 6)

    # ── Section 13: Engineered — Card Aggregation ──
    with st.expander("📊 Engineered: Card1 Aggregation", expanded=False):
        st.caption("Statistics of TransactionAmt grouped by card1 (from training set).")
        c1, c2, c3 = st.columns(3)
        with c1:
            values["card1_amt_mean"] = _float_slider("card1_amt_mean", "card1_amt_mean",
                                                      100.0, 0.0, 10000.0, 1.0)
        with c2:
            values["card1_amt_std"] = _float_slider("card1_amt_std", "card1_amt_std",
                                                     80.0, 0.0, 5000.0, 1.0)
        with c3:
            values["TransactionAmt_to_mean_card1"] = _float_slider("TransactionAmt / card1_amt_mean",
                                                                     "TransactionAmt_to_mean_card1",
                                                                     1.0, 0.0, 200.0, 0.01)

    # ── Section 14: Engineered — UID Aggregation ──
    with st.expander("👤 Engineered: UID Aggregation", expanded=False):
        st.caption("UID = card1 + addr1 + D1. Aggregation stats from training set.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            values["uid_count"] = _float_slider("uid_count", "uid_count", 3.0, 0.0, 5000.0, 1.0)
        with c2:
            values["uid_amt_mean"] = _float_slider("uid_amt_mean", "uid_amt_mean", 100.0, 0.0, 10000.0, 1.0)
        with c3:
            values["uid_amt_std"] = _float_slider("uid_amt_std", "uid_amt_std", 50.0, 0.0, 5000.0, 1.0)
        with c4:
            values["TransactionAmt_to_mean_uid"] = _float_slider("TransactionAmt / uid_amt_mean",
                                                                   "TransactionAmt_to_mean_uid",
                                                                   1.0, 0.0, 200.0, 0.01)

    # ── Section 15: Engineered — UID2 & Velocity ──
    with st.expander("⚡ Engineered: UID2 & Velocity", expanded=False):
        st.caption("UID2 = card1 + card2 + addr1 + P_emaildomain. Includes velocity (time since last txn).")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            values["uid2_count"] = _float_slider("uid2_count", "uid2_count", 2.0, 0.0, 5000.0, 1.0)
        with c2:
            values["uid2_amt_mean"] = _float_slider("uid2_amt_mean", "uid2_amt_mean", 100.0, 0.0, 10000.0, 1.0)
        with c3:
            values["TransactionAmt_to_mean_uid2"] = _float_slider("TransactionAmt / uid2_amt_mean",
                                                                    "TransactionAmt_to_mean_uid2",
                                                                    1.0, 0.0, 200.0, 0.01)
        with c4:
            values["time_since_last_uid_txn"] = _float_slider("time_since_last_uid_txn (sec, -1=first)",
                                                                "time_since_last_uid_txn",
                                                                -1.0, -1.0, 1000000.0, 100.0)

    # ── Predict Button ──
    st.write("---")
    col1, col2 = st.columns([1, 4])
    with col1:
        predict_btn = st.button("🔍 Analyze Transaction", type="primary", use_container_width=True)

    if predict_btn:
        if model is None:
            st.error("Model not loaded. Please ensure models/fraud_model.pkl exists.")
        else:
            try:
                # Build the feature array in the correct order
                feature_array = np.array([[values[f] for f in FEATURE_NAMES]])

                # Verify expected feature count if model attributes exist
                n_expected = getattr(model, "n_features_in_", getattr(model, "n_features_", len(FEATURE_NAMES)))
                if n_expected != len(FEATURE_NAMES):
                    st.warning(
                        f"⚠️ **Feature Mismatch Notice**: The current model file (`models/fraud_model.pkl`) expects "
                        f"**{n_expected}** features (type: `{type(model).__name__}`), but the full pipeline provides **{len(FEATURE_NAMES)}** features.\n\n"
                        f"Once your team exports the final 379-feature model from **Notebook 01**, overwrite `models/fraud_model.pkl` to test real predictions."
                    )
                else:
                    # Handle both LightGBM Booster (returns probabilities) and
                    # sklearn-style models (returns class labels)
                    raw_output = model.predict(feature_array)

                    if hasattr(model, "predict_proba"):
                        # sklearn-style: use predict directly
                        prediction = int(raw_output[0])
                        proba = float(model.predict_proba(feature_array)[0][1])
                    elif 0 <= raw_output[0] <= 1 and not isinstance(raw_output[0], (int, np.integer)):
                        # LightGBM Booster: output is a probability
                        proba = float(raw_output[0])
                        prediction = 1 if proba >= FRAUD_THRESHOLD else 0
                    else:
                        # Fallback: treat as class label
                        prediction = int(raw_output[0])
                        proba = None

                    st.write("---")
                    if prediction == 1:
                        st.error("🚨 ALERT: FRAUDULENT TRANSACTION DETECTED")
                        if proba is not None:
                            st.metric("Fraud Probability", f"{proba:.2%}", delta=f"Threshold: {FRAUD_THRESHOLD:.0%}")
                    else:
                        st.success("✅ STATUS: NORMAL LEGITIMATE TRANSACTION")
                        if proba is not None:
                            st.metric("Fraud Probability", f"{proba:.2%}", delta=f"Threshold: {FRAUD_THRESHOLD:.0%}",
                                      delta_color="off")
            except Exception as e:
                st.error(f"Error analyzing transaction data: {str(e)}")

# ─────────────────────────────────────────────────────────────────
# PAGE 3 — Project Info
# ─────────────────────────────────────────────────────────────────
elif page == "Project Info":
    st.markdown('<p class="main-header">Project Architecture</p>', unsafe_allow_html=True)

    st.info("Dataset: IEEE-CIS Fraud Detection from Kaggle")
    st.link_button("Dataset Link", "https://www.kaggle.com/competitions/ieee-fraud-detection", type="primary")
    st.write("### Workflow Stages")
    st.write("1. **Data Understanding & Cleaning**: Removing outliers & handling missing identities")
    st.write("2. **Feature Engineering**: Creating transactional aggregations & encoding categoricals")
    st.write("3. **Model Training**: Comparing XGBoost LightGBM & Random Forest with SMOTE")
    st.write("4. **Realtime Deployment**: RESTful API via FastAPI & interactive UI via Streamlit")

    st.write("### Model Details")
    if model is not None:
        st.write(f"- **Model type**: `{type(model).__name__}`")
        n_features = getattr(model, "n_features_in_", getattr(model, "n_features_", "N/A"))
        st.write(f"- **Expected features**: {n_features}")
        st.write(f"- **Fraud threshold**: {FRAUD_THRESHOLD}")
    else:
        st.warning("Model not loaded.")
