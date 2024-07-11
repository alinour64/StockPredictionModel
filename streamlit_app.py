import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

@st.cache_resource(show_spinner=False)
def load_prediction_model(model_path):
    try:
        model = load_model(model_path)
        st.success(f'Model loaded successfully from {model_path}')
        return model
    except Exception as e:
        st.error(f'Error loading model: {e}')
        return None

# Function to download stock data
def get_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    return data

# Function to scale data
def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    return scaled_data, scaler

# Function to prepare test dataset
def prepare_test_data(train_data, test_data):
    past_100_days = train_data.tail(100)
    extended_test_data = pd.concat([past_100_days, test_data], ignore_index=True)
    return extended_test_data

# Function to create datasets for prediction
def create_datasets(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Plotting function with Plotly
def plot_data(data_series, xlabel, ylabel, legends):
    fig = go.Figure()
    for series, label in zip(data_series, legends):
        fig.add_trace(go.Scatter(y=series, mode='lines', name=label))
    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)
    st.plotly_chart(fig, use_container_width=True)

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20):
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

# Streamlit interface setup
st.header('Stock Market Prediction Application')

# User input for stock ticker and date range
stock_ticker = st.text_input('Enter Stock Ticker', 'GOOG')
start_date = st.date_input('Start Date', datetime(2000, 1, 1))
end_date = st.date_input('End Date', datetime.today())

# Download stock data
stock_data = get_stock_data(stock_ticker, start_date, end_date)

# Display stock data
st.subheader('Stock Data')
st.write(stock_data)

# Train-test split
train_size = 0.8
train_data = stock_data['Close'][:int(len(stock_data) * train_size)]
test_data = stock_data['Close'][int(len(stock_data) * train_size):]

# Scale train data
scaled_train_data, scaler = scale_data(train_data)

# Prepare test data
extended_test_data = prepare_test_data(train_data, test_data)
scaled_test_data = scaler.transform(extended_test_data.values.reshape(-1, 1))

# Create datasets for model prediction
X_test, y_test = create_datasets(scaled_test_data, 100)

# Load the pre-trained model
model_path = 'Stock_Predictions_Model.keras'
model = load_prediction_model(model_path)

# Debugging: Check shape and type of X_test
st.sidebar.subheader('Debugging Information')
st.sidebar.write(f'X_test shape: {X_test.shape}')
st.sidebar.write(f'X_test dtype: {X_test.dtype}')

# Predict and plot results if the model is loaded
if model:
    try:
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Moving Averages
        ma_50 = stock_data['Close'].rolling(window=50).mean()
        ma_100 = stock_data['Close'].rolling(window=100).mean()
        ma_200 = stock_data['Close'].rolling(window=200).mean()

        # Plot Moving Averages
        st.subheader('Price vs 50-Day Moving Average')
        plot_data([stock_data['Close'], ma_50], 'Date', 'Price', ['Price', 'MA50'])

        st.subheader('Price vs 50-Day and 100-Day Moving Averages')
        plot_data([stock_data['Close'], ma_50, ma_100], 'Date', 'Price', ['Price', 'MA50', 'MA100'])

        st.subheader('Price vs 100-Day and 200-Day Moving Averages')
        plot_data([stock_data['Close'], ma_100, ma_200], 'Date', 'Price', ['Price', 'MA100', 'MA200'])

        # Plot original vs predicted prices
        st.subheader('Original Price vs Predicted Price')
        plot_data([y_test.flatten(), predictions.flatten()], 'Time (Number of Days)', 'Price', ['Original Price', 'Predicted Price'])

        # Additional Features
        st.subheader('Additional Features')

        # RSI Calculation
        stock_data['RSI'] = calculate_rsi(stock_data)

        # Bollinger Bands Calculation
        stock_data['Upper Band'], stock_data['Lower Band'] = calculate_bollinger_bands(stock_data)

        # Plot RSI
        st.subheader('Relative Strength Index (RSI)')
        plot_data([stock_data['RSI']], 'Date', 'RSI', ['RSI'])

        # Plot Bollinger Bands
        st.subheader('Bollinger Bands')
        plot_data([stock_data['Close'], stock_data['Upper Band'], stock_data['Lower Band']], 'Date', 'Price', ['Price', 'Upper Band', 'Lower Band'])

        # Display data with new features
        st.subheader('Data with RSI and Bollinger Bands')
        st.write(stock_data)
    except Exception as e:
        st.error(f'Error during model prediction: {e}')
else:
    st.error('Model could not be loaded. Please check the file path and model compatibility.')
