**Stock Market Prediction Application**

**Overview**

This application predicts stock prices and provides visual analysis using various technical indicators. It fetches historical stock data, scales it, prepares it for prediction, and visualizes the results along with indicators such as moving averages, RSI, and Bollinger Bands. The goal is to help users make informed decisions based on predicted stock trends and technical analysis.

**Tools Used**

1. **TensorFlow**:
    - An open-source platform for machine learning developed by Google. It is used for building and training machine learning models. In this application, TensorFlow is used to load and utilize the pre-trained prediction model.
2. **YFinance**:
    - A Python library that provides an easy way to download historical market data from Yahoo Finance. This library is used to fetch historical stock data for analysis and prediction.
3. **Plotly**:
    - An open-source graphing library that makes interactive, publication-quality graphs online. Plotly is used in this application to create interactive graphs and plots for data visualization.
4. **Numpy**:
    - A fundamental package for scientific computing in Python. It provides support for large multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays. Numpy is used for numerical operations and handling arrays in this application.
5. **Pandas**:
    - An open-source data analysis and manipulation tool built on top of the Python programming language. It is used for data manipulation and analysis, providing data structures like DataFrame for working with structured data.
6. **Streamlit**:
    - An open-source app framework for Machine Learning and Data Science teams to create beautiful, performant apps in hours, not weeks. Streamlit is used for creating the web application and interactive user interface.
7. **Scikit-learn (MinMaxScaler)**:
    - A machine learning library for Python that provides simple and efficient tools for data mining and data analysis. MinMaxScaler from Scikit-learn is used to scale data to a range between 0 and 1.

**Program Features**

1. **Stock Data Retrieval**: Fetches historical stock data for a specified ticker and date range using YFinance.
2. **Data Preparation and Scaling**: Splits the data into training and testing sets, scales the data using MinMaxScaler, and prepares it for prediction.
    - **Training and Testing Split**: 80% of the data is used for training, and 20% is used for testing.
3. **Model Loading and Prediction**: Loads a pre-trained TensorFlow model and makes predictions on the test data.
4. **Visualization**: Provides interactive visualizations of stock prices, predictions, and technical indicators such as moving averages, RSI, and Bollinger Bands using Plotly.
5. **Technical Indicators**: Calculates and displays Relative Strength Index (RSI) and Bollinger Bands.

**Development Process**

1. **Model Training**:
    - The prediction model was trained using Jupyter Notebooks.
    - Data was split into 80% for training and 20% for testing to ensure the model was trained and evaluated effectively.
2. **Application Integration**:
    - The trained model and data processing steps were integrated into a Streamlit application for an interactive user experience.
    - The application was designed to be user-friendly, allowing users to input stock tickers and date ranges to fetch and analyze data.
3. **Visualization**:
    - Interactive visualizations were implemented using Plotly to enable users to interact with the data and gain insights.
    - Key visualizations include original vs. predicted prices, moving averages, RSI, and Bollinger Bands.
4. **Deployment**:
    - The application was deployed on Streamlit.
    - A requirements.txt file was created to ensure all dependencies were met for smooth operation.
    - The deployment process involved setting up the application to run efficiently on Streamlit, ensuring accessibility and ease of use.

**Challenges and Solutions**

1. **Training the Model**:
    - The model was trained using Jupyter Notebooks and later adapted for integration with the Streamlit application.
    - Ensuring the model performed well required careful data splitting and preprocessing.
2. **Interactive Visualization**:
    - Plotly was selected over Matplotlib for its superior interactive capabilities, enhancing the user experience.
    - Creating interactive plots required restructuring the data and implementing dynamic plot updates.
3. **Deployment**:
    - The application was successfully deployed on Streamlit, with careful attention to dependency management to ensure compatibility and functionality.
    - Creating the requirements.txt file and configuring the environment were key steps to ensure the application ran smoothly on Streamlit.

**Conclusion**

This application provides a comprehensive solution for stock market prediction and analysis. By combining machine learning, data visualization, and an interactive web interface, it enables users to make informed decisions based on predicted stock trends and technical analysis. The use of TensorFlow, YFinance, Plotly, and Streamlit ensures the application is robust, scalable, and user-friendly.
