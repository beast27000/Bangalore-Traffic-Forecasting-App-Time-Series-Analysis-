import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit as st
import pickle
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Paths
DATA_PATH = r"C:\Advanced projects\Time series models\cleaned_traffic_time_series_data.csv"
MODEL_PATH = r"C:\Advanced projects\Time series models\traffic_volume_model.pkl"

# Load dataset
if os.path.exists(DATA_PATH):
    data = pd.read_csv(DATA_PATH)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
else:
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

# Train SARIMA model
def train_model(data):
    target_series = data['Traffic_Volume']
    model = SARIMAX(target_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24), enforce_stationarity=False)
    results = model.fit(disp=False)
    return results

# Save model
if not os.path.exists(MODEL_PATH):
    st.write("Training the SARIMA model...")
    sarima_results = train_model(data)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(sarima_results, f)
    st.success("Model trained and saved successfully!")
else:
    st.write("Loading existing model...")
    with open(MODEL_PATH, 'rb') as f:
        sarima_results = pickle.load(f)

# Forecast traffic volume for the next year (hourly)
forecast_steps = 365 * 24  # Forecast for every hour in the next year
forecast = sarima_results.get_forecast(steps=forecast_steps)
forecast_df = forecast.summary_frame()
forecast_df['DateTime'] = pd.date_range(start=data.index[-1] + pd.Timedelta(hours=1), periods=forecast_steps, freq='H')
forecast_df['Date'] = forecast_df['DateTime'].dt.date
forecast_df['Hour'] = forecast_df['DateTime'].dt.hour
forecast_df.set_index('DateTime', inplace=True)

# Calculate evaluation metrics
def calculate_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    return rmse, r2

# Generate heatmap
def generate_heatmap(data):
    daily_data = data.resample('D').mean()
    heatmap_data = daily_data.pivot_table(index=daily_data.index.month, columns=daily_data.index.day, values='Traffic_Volume')
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap="YlGnBu", annot=False, cbar=True)
    plt.title("Traffic Volume Heatmap")
    plt.xlabel("Day")
    plt.ylabel("Month")
    st.pyplot(plt)

# Streamlit App
st.title("Bangalore Traffic Volume Prediction App")

# Display evaluation metrics
st.subheader("Model Evaluation Metrics")
train_data = data['Traffic_Volume'][:-forecast_steps]
test_data = data['Traffic_Volume'][-forecast_steps:]
rmse, r2 = calculate_metrics(test_data, forecast_df['mean'][:len(test_data)])
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
st.write(f"R-squared (R²): {r2:.2f}")

# Heatmap
st.subheader("Traffic Volume Heatmap")
generate_heatmap(data)

# Forecast visualization
st.subheader("Predicted Traffic Volume for the Next Year")
traffic_forecast = forecast_df[['mean']]
st.line_chart(traffic_forecast)

# Option to download the forecast
csv = forecast_df.to_csv(index=True)
st.download_button("Download Traffic Volume Prediction", data=csv, file_name="traffic_volume_forecast.csv", mime="text/csv")
