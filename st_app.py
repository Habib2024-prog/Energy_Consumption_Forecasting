import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle
import warnings

warnings.filterwarnings("ignore")

# ==============================
# SAFE TENSORFLOW IMPORT (IMPORTANT)
# ==============================
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False
    load_model = None


# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Energy Consumption Forecasting",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Energy Consumption Forecasting Dashboard")


if not TF_AVAILABLE:
    st.warning("LSTM model disabled (TensorFlow not available), but app works fully.")


# ==============================
# LOAD FILES
# ==============================
@st.cache_resource
def load_models():
    scaler_X = pickle.load(open("models/scaler_X.pkl", "rb"))
    scaler_Y = pickle.load(open("models/scaler_Y.pkl", "rb"))
    feature_cols = pickle.load(open("models/feature_cols.pkl", "rb"))

    ridge = pickle.load(open("models/ridge_model.pkl", "rb"))
    rf = pickle.load(open("models/rf_model.pkl", "rb"))

    lstm = None
    if TF_AVAILABLE:
        try:
            lstm = load_model("models/lstm_model.keras")
        except:
            lstm = None

    return scaler_X, scaler_Y, feature_cols, ridge, rf, lstm


@st.cache_data
def load_data():
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    df = pd.concat([train, test]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df, train, test


scaler_X, scaler_Y, feature_cols, ridge, rf, lstm = load_models()
df, train_df, test_df = load_data()


# ==============================
# SIDEBAR FILTERS
# ==============================
st.sidebar.header("Filters")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "Ridge Regression", "All Models"]
)

test_dates = pd.to_datetime(test_df["date"])
min_date = test_dates.min().date()
max_date = test_dates.max().date()

start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

n_points = st.sidebar.slider("Points to Show", 100, 1000, 400)


# ==============================
# FILTER DATA
# ==============================
mask = (
    (pd.to_datetime(test_df["date"]).dt.date >= start_date) &
    (pd.to_datetime(test_df["date"]).dt.date <= end_date)
)

filtered = test_df[mask]

X_test = scaler_X.transform(filtered[feature_cols])
y_test = scaler_Y.transform(filtered[["Appliances"]])
y_true = scaler_Y.inverse_transform(y_test).ravel()


# ==============================
# TABS (YOUR ORIGINAL STRUCTURE)
# ==============================
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🔮 Predictions", "📁 Dataset Info"])


# =========================================================
# TAB 1 — EDA
# =========================================================
with tab1:
    st.subheader("📊 Exploratory Data Analysis")

    # Line plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["Appliances"],
        mode="lines",
        name="Energy Usage"
    ))

    fig.update_layout(
        title="Energy Consumption Over Time",
        xaxis_title="Date",
        yaxis_title="Energy (Wh)"
    )

    st.plotly_chart(fig, use_container_width=True)


    # Hourly pattern
    st.subheader("⏰ Hourly Pattern")

    df["hour"] = df["date"].dt.hour
    hourly = df.groupby("hour")["Appliances"].mean().reset_index()

    fig2 = px.bar(hourly, x="hour", y="Appliances")
    st.plotly_chart(fig2, use_container_width=True)


    # Distribution
    st.subheader("📦 Distribution")

    fig3 = px.histogram(df, x="Appliances", nbins=50)
    st.plotly_chart(fig3, use_container_width=True)


# =========================================================
# TAB 2 — PREDICTIONS
# =========================================================
with tab2:
    st.subheader("🔮 Model Predictions vs Actual")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=y_true[:n_points],
        mode="lines",
        name="Actual"
    ))

    # RF
    if model_choice in ["Random Forest", "All Models"]:
        rf_pred = rf.predict(X_test)
        rf_pred = scaler_Y.inverse_transform(rf_pred.reshape(-1, 1)).ravel()

        fig.add_trace(go.Scatter(
            y=rf_pred[:n_points],
            mode="lines",
            name="Random Forest"
        ))

    # Ridge
    if model_choice in ["Ridge Regression", "All Models"]:
        ridge_pred = ridge.predict(X_test)
        ridge_pred = scaler_Y.inverse_transform(ridge_pred.reshape(-1, 1)).ravel()

        fig.add_trace(go.Scatter(
            y=ridge_pred[:n_points],
            mode="lines",
            name="Ridge"
        ))

    st.plotly_chart(fig, use_container_width=True)


    # Metrics
    st.subheader("📌 Metrics")

    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    metrics = {}

    rf_pred = rf.predict(X_test)
    rf_pred = scaler_Y.inverse_transform(rf_pred.reshape(-1, 1)).ravel()
    metrics["Random Forest RMSE"] = rmse(y_true, rf_pred)

    ridge_pred = ridge.predict(X_test)
    ridge_pred = scaler_Y.inverse_transform(ridge_pred.reshape(-1, 1)).ravel()
    metrics["Ridge RMSE"] = rmse(y_true, ridge_pred)

    st.write(metrics)


# =========================================================
# TAB 3 — DATASET INFO
# =========================================================
with tab3:
    st.subheader("📁 Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))
    col2.metric("Features", len(feature_cols))
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.write(df.describe())

    st.subheader(" Sample Data")
    st.dataframe(df.head(50))