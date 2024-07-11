**Stock Market Prediction Application**



**Overview**

This application predicts stock prices and provides visual analysis using various technical indicators. The application fetches historical stock data, scales it, prepares it for prediction, and visualizes the results along with technical indicators such as moving averages, RSI, and Bollinger Bands. The goal is to help users make informed decisions based on the predicted stock trends and technical analysis.
**Tools Used**

1.	TensorFlow: For loading and utilizing the pre-trained prediction model.
2.	YFinance: For fetching historical stock data.
3.	Plotly: For creating interactive graphs and plots.
4.	Numpy: For numerical operations and handling arrays.
5.	Pandas: For data manipulation and analysis.
6.	Streamlit: For creating the web application and interactive UI.
**Program Features**

1.	Stock Data Retrieval: Fetches historical stock data for a specified ticker and date range.
2.	Data Preparation and Scaling: Splits the data into training and testing sets, scales the data using MinMaxScaler, and prepares it for prediction.
3.	Model Loading and Prediction: Loads a pre-trained TensorFlow model and makes predictions on the test data.
4.	Visualization: Provides interactive visualizations of stock prices, predictions, and technical indicators such as moving averages, RSI, and Bollinger Bands using Plotly.
5.	Technical Indicators: Calculates and displays Relative Strength Index (RSI) and Bollinger Bands.
**Development Process**

1.	Model Training: The prediction model was initially trained using Jupyter Notebooks.
2.	Application Integration: The trained model and data processing steps were integrated into a Streamlit application for an interactive user experience.
3.	Visualization: Interactive visualizations were implemented using Plotly to enable users to interact with the data and gain insights.
4.	Deployment: The application was deployed on Streamlit, with a requirements.txt file created to ensure all dependencies are met for smooth operation.
**Challenges and Solutions**

1.	Training the Model: The model was trained using Jupyter Notebooks and later adapted for integration with the Streamlit application.
2.	Interactive Visualization: Plotly was selected over Matplotlib for its superior interactive capabilities, enhancing the user experience.
3.	Deployment: The application was successfully deployed on Streamlit, with careful attention to dependency management to ensure compatibility and functionality.

