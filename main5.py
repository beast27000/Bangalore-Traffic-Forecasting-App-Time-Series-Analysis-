#   THE BELOW CODE GIVES THE HOURLY  VALUES BASES FOR DAILY OR MONTHLY TRAFFIC PRDICTION FOR THE NEXT ONE YEAR 


import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit as st
import pickle
import os

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

# Streamlit App
st.title("Bangalore Traffic Volume Prediction App")

st.subheader("Predicted Traffic Volume for the Next Year")
traffic_forecast = forecast_df[['mean']]
st.line_chart(traffic_forecast)

# Interactive exploration
st.subheader("Explore Predictions")
time_unit = st.selectbox("Select Time Unit", ["Days", "Months"])
if time_unit == "Days":
    days = st.slider("Select number of days to view", 1, 365, 30)
    st.write(f"Predicted Traffic Volume for the next {days} days:")
    daily_data = traffic_forecast.resample('D').mean()
    st.line_chart(daily_data[:days])
    st.dataframe(daily_data[:days].rename(columns={"mean": "Predicted Traffic Volume"}))
elif time_unit == "Months":
    months = st.slider("Select number of months to view", 1, 12, 3)
    days = months * 30
    st.write(f"Predicted Traffic Volume for the next {months} months:")
    monthly_data = traffic_forecast.resample('M').mean()
    st.line_chart(monthly_data[:months])
    st.dataframe(monthly_data[:months].rename(columns={"mean": "Predicted Traffic Volume"}))

# Separate button for time-based exploration
st.subheader("Explore Predictions by Specific Time")
if st.button("View Hourly Traffic Data"):
    selected_date = st.date_input("Select Date", value=pd.Timestamp(data.index[-1] + pd.Timedelta(days=1)))
    selected_hour = st.slider("Select Hour", 0, 23, 12)
    filtered_data = forecast_df[(forecast_df['Date'] == pd.Timestamp(selected_date).date()) &
                                (forecast_df['Hour'] == selected_hour)]
    if not filtered_data.empty:
        st.write(f"Predicted Traffic Volume for {selected_date} at {selected_hour}:00")
        st.dataframe(filtered_data[['mean']].rename(columns={"mean": "Predicted Traffic Volume"}))
    else:
        st.warning("No data available for the selected date and time.")

# Option to download the forecast
csv = forecast_df.to_csv(index=True)
st.download_button("Download Traffic Volume Prediction", data=csv, file_name="traffic_volume_forecast.csv", mime="text/csv")
