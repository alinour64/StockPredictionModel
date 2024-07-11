import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import talib

# User Authentication
def login_user(username, password):
    return username == "admin" and password == "password"

st.sidebar.title("Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type='password')
login_button = st.sidebar.button("Login")

if login_button:
    if login_user(username, password):
        st.sidebar.success("Login successful!")
    else:
        st.sidebar.error("Invalid username or password")
        st.stop()

# Load the pre-trained model with error handling
model_path = 'Stock_Predictions_Model.keras'
try:
    model = load_model(model_path)
    st.success(f'Model loaded successfully from {model_path}')
except Exception as e:
    st.error(f'Error loading model: {e}')

# Set up Streamlit
st.header('Stock Market Prediction Application')

# User input for stock ticker and date range
stock_ticker = st.text_input('Enter Stock Ticker', 'GOOG')
start_date = st.date_input('Start Date', datetime(2000, 1, 1))
end_date = st.date_input('End Date', datetime.today())

# Download stock data from Yahoo Finance
stock_data = yf.download(stock_ticker, start=start_date, end=end_date)

# Display stock data
st.subheader('Stock Data')
st.write(stock_data)

# Train-test split
train_size = 0.8
train_data = stock_data['Close'][:int(len(stock_data) * train_size)]
test_data = stock_data['Close'][int(len(stock_data) * train_size):]

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data.values.reshape(-1, 1))

# Prepare test data
past_100_days = train_data.tail(100)
extended_test_data = pd.concat([past_100_days, test_data], ignore_index=True)
scaled_test_data = scaler.transform(extended_test_data.values.reshape(-1, 1))

# Create datasets for model prediction
X_test, y_test = [], []
for i in range(100, len(scaled_test_data)):
    X_test.append(scaled_test_data[i-100:i])
    y_test.append(scaled_test_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

# Predict using the loaded model
if 'model' in locals():
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plotting function
    def plot_interactive_data(data_series, title, xlabel, ylabel, legends):
        fig = go.Figure()
        for series, label in zip(data_series, legends):
            fig.add_trace(go.Scatter(x=series.index, y=series, mode='lines', name=label))
        fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
        st.plotly_chart(fig)

    # Moving Averages
    ma_50 = stock_data['Close'].rolling(window=50).mean()
    ma_100 = stock_data['Close'].rolling(window=100).mean()
    ma_200 = stock_data['Close'].rolling(window=200).mean()

    # Plot Moving Averages
    st.subheader('Price vs 50-Day Moving Average')
    plot_interactive_data([stock_data['Close'], ma_50], 'Stock Price vs 50-Day Moving Average', 'Date', 'Price', ['Price', 'MA50'])

    st.subheader('Price vs 50-Day and 100-Day Moving Averages')
    plot_interactive_data([stock_data['Close'], ma_50, ma_100], 'Stock Price vs 50-Day and 100-Day Moving Averages', 'Date', 'Price', ['Price', 'MA50', 'MA100'])

    st.subheader('Price vs 100-Day and 200-Day Moving Averages')
    plot_interactive_data([stock_data['Close'], ma_100, ma_200], 'Stock Price vs 100-Day and 200-Day Moving Averages', 'Date', 'Price', ['Price', 'MA100', 'MA200'])

    st.subheader('Original Price vs Predicted Price')
    plot_interactive_data([pd.Series(y_test.flatten(), index=test_data.index), pd.Series(predictions.flatten(), index=test_data.index)], 'Original vs Predicted Stock Prices', 'Time (Number of Days)', 'Price', ['Original Price', 'Predicted Price'])

    # Additional Features
    st.subheader('Additional Features')

    # RSI Calculation
    def calculate_rsi(data, window=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    stock_data['RSI'] = calculate_rsi(stock_data)

    # MACD Calculation
    stock_data['MACD'], stock_data['MACD_Signal'], _ = talib.MACD(stock_data['Close'])
    
    # Stochastic Oscillator Calculation
    stock_data['Slowk'], stock_data['Slowd'] = talib.STOCH(stock_data['High'], stock_data['Low'], stock_data['Close'])

    # Bollinger Bands Calculation
    def calculate_bollinger_bands(data, window=20):
        sma = data['Close'].rolling(window=window).mean()
        std = data['Close'].rolling(window=window).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return upper_band, lower_band

    stock_data['Upper Band'], stock_data['Lower Band'] = calculate_bollinger_bands(stock_data)

    # Plot RSI
    st.subheader('Relative Strength Index (RSI)')
    plot_interactive_data([stock_data['RSI']], 'Relative Strength Index (RSI)', 'Date', 'RSI', ['RSI'])

    # Plot MACD
    st.subheader('MACD')
    plot_interactive_data([stock_data['MACD'], stock_data['MACD_Signal']], 'MACD', 'Date', 'Value', ['MACD', 'Signal'])

    # Plot Stochastic Oscillator
    st.subheader('Stochastic Oscillator')
    plot_interactive_data([stock_data['Slowk'], stock_data['Slowd']], 'Stochastic Oscillator', 'Date', 'Value', ['Slowk', 'Slowd'])

    # Plot Bollinger Bands
    st.subheader('Bollinger Bands')
    plot_interactive_data([stock_data['Close'], stock_data['Upper Band'], stock_data['Lower Band']], 'Bollinger Bands', 'Date', 'Price', ['Price', 'Upper Band', 'Lower Band'])

    # Display data with new features
    st.subheader('Data with Technical Indicators')
    st.write(stock_data)
else:
    st.error('Model could not be loaded. Please check the file path and model compatibility.')
