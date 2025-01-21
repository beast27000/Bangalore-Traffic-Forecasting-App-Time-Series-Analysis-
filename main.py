import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import LabelEncoder
import pickle
import streamlit as st
import os

# Paths
MODEL_PATH = r"C:\\Advanced projects\\Bangalore_Traffic\\traffic_time_series_model.pkl"
ENCODER_PATH = r"C:\\Advanced projects\\Bangalore_Traffic\\label_encoders.pkl"
CLEANED_DATA_PATH = "cleaned_traffic_time_series_data.csv"

# Ensure directory exists
if not os.path.exists(os.path.dirname(MODEL_PATH)):
    os.makedirs(os.path.dirname(MODEL_PATH))

# Load the dataset
data = pd.read_csv("C:/Advanced projects/Bangalore_Traffic/Banglore_traffic_Dataset.csv")

# Data Preprocessing
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y', errors='coerce')
data = data.sort_values(by='Date')
data.set_index('Date', inplace=True)

# Handle categorical columns by encoding them
categorical_columns = ['Area Name', 'Road/Intersection Name', 'Weather Conditions',
                       'Traffic Signal Compliance', 'Roadwork and Construction Activity']
encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Save the encoders to a file
with open(ENCODER_PATH, 'wb') as f:
    pickle.dump(encoders, f)

# Fill missing values
data = data.fillna(data.mean())
for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# Save cleaned data for reusability
data.to_csv(CLEANED_DATA_PATH)

# Time Series Modeling with Statsmodels
st.title("Bangalore Traffic Time Series Forecasting App")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    input_data['Date'] = pd.to_datetime(input_data['Date'], format='%d-%m-%Y', errors='coerce')
    input_data.set_index('Date', inplace=True)
    st.write("Input Data:")
    st.write(input_data)

    # Forecasting Traffic Volume
    st.subheader("Traffic Volume Forecasting")
    target_series = data['Traffic_Volume']

    # Fit a SARIMA model
    st.write("Fitting SARIMA model...")
    sarima_model = SARIMAX(target_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False)
    sarima_results = sarima_model.fit(disp=False)

    # Save the model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(sarima_results, f)

    # Forecast future values
    forecast_steps = st.slider("Select number of days to forecast", 1, 30, 7)
    forecast = sarima_results.get_forecast(steps=forecast_steps)
    forecast_df = forecast.summary_frame()
    st.write("Forecasted Traffic Volume:")
    st.write(forecast_df)

    # Plot the forecast
    st.line_chart(forecast_df['mean'])

    # Option to download the forecast
    csv = forecast_df.to_csv(index=True)
    st.download_button("Download Forecast", data=csv, file_name="traffic_forecast.csv", mime="text/csv")
