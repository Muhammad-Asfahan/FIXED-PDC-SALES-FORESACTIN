# ============================================================
# ADVANCED SALES DEMAND PREDICTION SYSTEM (PRO VERSION)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Sales AI Dashboard",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# PROFESSIONAL UI DESIGN (DARK SaaS STYLE)
# ============================================================

st.markdown("""
<style>

/* ========================= GLOBAL ========================= */

* {
    font-family: 'Segoe UI', sans-serif;
}

/* ========================= BACKGROUND ========================= */

.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: #e2e8f0;
}

/* ========================= TITLE ========================= */

.main-title {
    font-size: 54px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #38bdf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 20px;
}

/* ========================= TEXT ========================= */

h1, h2, h3 {
    color: #f8fafc !important;
    font-weight: 700;
}

p, span, label {
    color: #cbd5e1 !important;
}

/* ========================= SIDEBAR ========================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1220, #111827);
}

section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* ========================= BUTTONS ========================= */

.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 14px;
    border: none;
    font-size: 17px;
    font-weight: 700;
    color: white;
    background: linear-gradient(135deg, #06b6d4, #7c3aed);
    box-shadow: 0px 4px 20px rgba(124,58,237,0.4);
    transition: 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-3px);
    background: linear-gradient(135deg, #3b82f6, #a855f7);
}

/* ========================= INPUTS ========================= */

input, textarea {
    background-color: #0f172a !important;
    color: white !important;
}

/* ========================= METRICS ========================= */

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 18px;
    padding: 15px;
    backdrop-filter: blur(10px);
}

/* ========================= DATAFRAME ========================= */

[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.03);
    border-radius: 15px;
}

/* ========================= CHARTS ========================= */

.js-plotly-plot {
    background: rgba(255,255,255,0.03);
    border-radius: 20px;
    padding: 10px;
}

/* ========================= SCROLLBAR ========================= */

::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-thumb {
    background: #7c3aed;
    border-radius: 10px;
}

::-webkit-scrollbar-track {
    background: #0f172a;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class='main-title'>
📊 Sales Intelligence AI Dashboard
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sales_prediction_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "model_features.pkl")
DATA_PATH = os.path.join(BASE_DIR, "processed_superstore.csv")

model = joblib.load(MODEL_PATH)
model_features = joblib.load(FEATURES_PATH)
df = pd.read_csv(DATA_PATH)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📌 Control Panel")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📈 Prediction", "📊 Analytics", "📋 Dataset"]
)

# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.title("AI Powered Sales Forecast System")

    st.markdown("""
    ### 🚀 Features
    - Smart Sales Prediction
    - Advanced Analytics
    - AI-Based Forecasting
    - Interactive Dashboard

    ### ⚙️ Tech Stack
    Streamlit • Plotly • Scikit-learn • Pandas
    """)

# ============================================================
# PREDICTION
# ============================================================

elif page == "📈 Prediction":

    st.title("Sales Prediction Engine")

    col1, col2 = st.columns(2)

    with col1:
        quantity = st.number_input("Quantity", 0, 100, 1)
        discount = st.slider("Discount", 0.0, 1.0, 0.0)
        profit = st.number_input("Profit", 0.0)

    with col2:
        region = st.selectbox("Region", [0,1,2,3])
        category = st.selectbox("Category", [0,1,2])
        ship_mode = st.selectbox("Ship Mode", [0,1,2,3])

    if st.button("Predict Sales"):

        input_df = pd.DataFrame([{
            "Quantity": quantity,
            "Discount": discount,
            "Profit": profit,
            "Region": region,
            "Category": category,
            "Ship Mode": ship_mode
        }])

        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[model_features]

        pred = model.predict(input_df)[0]

        st.success(f"Predicted Sales: {pred:.2f}")

        # ===================== MODERN GAUGE =====================

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            title={'text': "Sales Forecast"},
            gauge={
                'axis': {'range': [0, max(1000, pred+100)]},
                'bar': {'color': "#06b6d4"},
                'steps': [
                    {'range': [0, 300], 'color': "#1e293b"},
                    {'range': [300, 700], 'color': "#334155"},
                    {'range': [700, 1200], 'color': "#0ea5e9"}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.title("Business Intelligence Dashboard")

    st.metric("Total Sales", f"{df['sales'].sum():,.0f}")
    st.metric("Avg Sales", f"{df['sales'].mean():,.0f}")

    # ===================== COLORFUL HISTOGRAM =====================

    fig1 = px.histogram(
        df,
        x="sales",
        nbins=40,
        color_discrete_sequence=["#7c3aed"]
    )

    fig1.update_layout(
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font_color="white"
    )

    st.plotly_chart(fig1, use_container_width=True)

    # ===================== HEATMAP =====================

    corr = df.select_dtypes(include=np.number).corr()

    fig2 = px.imshow(
        corr,
        color_continuous_scale="Turbo"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ===================== FEATURE IMPORTANCE =====================

    imp = pd.DataFrame({
        "Feature": model_features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig3 = px.bar(
        imp.head(10),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Viridis"
    )

    st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# DATASET
# ============================================================

elif page == "📋 Dataset":

    st.title("Dataset Overview")

    st.dataframe(df.head())

    st.write("Shape:", df.shape)

    st.write("Columns:", df.columns.tolist())

    st.write(df.describe())