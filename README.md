Stock Market Prediction Application
This application predicts stock prices and provides visual analysis using various technical indicators. It is built with several tools and libraries, including TensorFlow, YFinance, Plotly, and Streamlit. The application fetches historical stock data, scales it, prepares it for prediction, and visualizes the results along with technical indicators such as moving averages, RSI, and Bollinger Bands.

Tools Used
TensorFlow: Used for loading the pre-trained prediction model.
YFinance: Utilized for fetching historical stock data.
Plotly: Replaced Matplotlib for creating interactive graphs and plots.
Numpy: Used for numerical operations and handling arrays.
Pandas: Used for data manipulation and analysis.
Streamlit: Used for creating the web application and interactive UI.
Installation
Clone the repository.
git clone <repository-url>
cd <repository-directory>

Create a virtual environment and activate it.
python3 -m venv venv
source venv/bin/activate

Install the required dependencies.
pip install -r requirements.txt

Usage
Run the Streamlit application.
streamlit run app.py
Open your web browser and go to http://localhost:8501 to view the application.

Application Workflow
Header and User Inputs:
The application begins with a header and allows the user to input a stock ticker, start date, and end date.

Fetching Stock Data:
The application fetches historical stock data for the specified ticker and date range using YFinance.

Displaying Stock Data:
The fetched stock data is displayed on the Streamlit interface for user review.

Data Preparation and Scaling:
The application prepares the data by splitting it into training and testing sets, then scales the data using MinMaxScaler.

Loading the Prediction Model:
The pre-trained prediction model is loaded from a specified path.

Making Predictions:
If the model is successfully loaded, the application makes predictions on the test data.

Visualizing Results:
The application visualizes the original and predicted prices along with various technical indicators such as moving averages, RSI, and Bollinger Bands using Plotly for interactive plots.

Features
Stock Data Retrieval: Fetches and displays historical stock data.
Data Scaling and Preparation: Scales data and prepares it for prediction.
Model Loading and Prediction: Loads a pre-trained TensorFlow model and makes predictions on the test data.
Interactive Plots: Provides interactive visualizations of stock prices, predictions, and technical indicators.
Technical Indicators: Calculates and displays RSI and Bollinger Bands.
Challenges and Solutions
Training the Model:

The model was initially trained using Jupyter Notebooks and later integrated into the Streamlit application.
Interactive Visualization:

Plotly was chosen over Matplotlib to enable better interactive capabilities for data visualization.
Deployment:

The application was deployed on Streamlit. A requirements.txt file was created to ensure all dependencies are met for smooth deployment.
Development Process
Model Training: Developed and trained the prediction model using Jupyter Notebooks.
Application Integration: Integrated the model and data processing steps into a Streamlit application.
Visualization: Implemented interactive visualizations using Plotly.
Deployment: Deployed the application on Streamlit, ensuring all dependencies were correctly specified.
