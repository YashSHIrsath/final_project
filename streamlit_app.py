
import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="⚡ Electricity Demand Forecast", layout="wide")
st.title("⚡ GB Electricity Demand Forecasting")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("historic_demand_2009_2024.csv")
    df.columns = df.columns.str.lower()
    df = df[df["settlement_period"] <= 48]
    df["datetime"] = pd.to_datetime(df["settlement_date"]) + pd.to_timedelta((df["settlement_period"] - 1) * 30, unit="min")
    df = df[["datetime", "nd"]].rename(columns={"datetime": "ds", "nd": "y"})
    return df

df = load_data()

# Show recent data
st.subheader("📊 Historical Demand")
st.line_chart(df.set_index("ds").tail(1000))

# Forecast controls
st.sidebar.header("🔧 Forecast Settings")
periods = st.sidebar.slider("Forecast Days", 1, 30, 7)

# Train model
model = Prophet()
model.fit(df)

# Create future dataframe
future = model.make_future_dataframe(periods=periods * 48, freq="30min")
forecast = model.predict(future)

# Plot forecast
st.subheader("🔮 Forecasted Demand")
fig1 = model.plot(forecast)
st.pyplot(fig1)

# Show forecast data
st.subheader("🧾 Forecast Table (Last 10)")
st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10))
