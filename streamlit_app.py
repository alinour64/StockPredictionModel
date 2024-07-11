import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


def load_prediction_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f'Error loading model: {e}')
        return None

def get_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    return data

def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    return scaled_data, scaler

def prepare_test_data(train_data, test_data):
    past_100_days = train_data.tail(100)
    extended_test_data = pd.concat([past_100_days, test_data], ignore_index=True)
    return extended_test_data

def create_datasets(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def plot_data(data_series, xlabel, ylabel, legends):
    fig = go.Figure()
    for series, label in zip(data_series, legends):
        fig.add_trace(go.Scatter(x=series.index, y=series, mode='lines', name=label))
    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)
    st.plotly_chart(fig)

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(data, window=20):
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

st.header('Stock Market Prediction Application')

stock_ticker = st.text_input('Enter Stock Ticker', 'GOOG')
start_date = st.date_input('Start Date', datetime(2000, 1, 1))
end_date = st.date_input('End Date', datetime.today())

stock_data = get_stock_data(stock_ticker, start_date, end_date)

st.subheader('Stock Data')
st.write(stock_data)

train_size = 0.8
train_data = stock_data['Close'][:int(len(stock_data) * train_size)]
test_data = stock_data['Close'][int(len(stock_data) * train_size):]

scaled_train_data, scaler = scale_data(train_data)

extended_test_data = prepare_test_data(train_data, test_data)
scaled_test_data = scaler.transform(extended_test_data.values.reshape(-1, 1))

X_test, y_test = create_datasets(scaled_test_data, 100)

model_path = 'Stock_Predictions_Model.keras'
model = load_prediction_model(model_path)

st.sidebar.subheader('Debugging Information')
st.sidebar.write(f'X_test shape: {X_test.shape}')
st.sidebar.write(f'X_test dtype: {X_test.dtype}')

if model:
    try:
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        ma_50 = stock_data['Close'].rolling(window=50).mean()
        ma_100 = stock_data['Close'].rolling(window=100).mean()
        ma_200 = stock_data['Close'].rolling(window=200).mean()

        st.subheader('Price vs 50-Day Moving Average')
        plot_data([stock_data['Close'], ma_50], 'Date', 'Price', ['Price', 'MA50'])

        st.subheader('Price vs 50-Day and 100-Day Moving Averages')
        plot_data([stock_data['Close'], ma_50, ma_100], 'Date', 'Price', ['Price', 'MA50', 'MA100'])

        st.subheader('Price vs 100-Day and 200-Day Moving Averages')
        plot_data([stock_data['Close'], ma_100, ma_200], 'Date', 'Price', ['Price', 'MA100', 'MA200'])

        st.subheader('Original Price vs Predicted Price')
        plot_data([pd.Series(y_test.flatten(), index=test_data.index), pd.Series(predictions.flatten(), index=test_data.index)], 'Date', 'Price', ['Original Price', 'Predicted Price'])

        st.subheader('Additional Features')

        stock_data['RSI'] = calculate_rsi(stock_data)

        stock_data['Upper Band'], stock_data['Lower Band'] = calculate_bollinger_bands(stock_data)

        st.subheader('Relative Strength Index (RSI)')
        plot_data([stock_data['RSI']], 'Date', 'RSI', ['RSI'])

        st.subheader('Bollinger Bands')
        plot_data([stock_data['Close'], stock_data['Upper Band'], stock_data['Lower Band']], 'Date', 'Price', ['Price', 'Upper Band', 'Lower Band'])

        st.subheader('Data with RSI and Bollinger Bands')
        st.write(stock_data)
    except Exception as e:
        st.error(f'Error during model prediction: {e}')
else:
    st.error('Model could not be loaded. Please check the file path and model compatibility.')
