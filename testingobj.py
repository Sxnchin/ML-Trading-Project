import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import yfinance as yf
import matplotlib.pyplot as plt

# Step 1: Data Collection
def get_stock_data(ticker, start_date, end_date):
    print(f"Fetching stock data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    print("Data fetched successfully.")
    print("Columns in the data:", data.columns)  # Print the columns to debug
    print("First few rows of the data:\n", data.head())  # Print the first few rows to debug
    if ('Close', ticker) in data.columns:
        data['Return'] = data[('Close', ticker)].pct_change()
    else:
        print(f"Column 'Close' for ticker {ticker} not found. Available columns:", data.columns)
        data.to_csv('fetched_data.csv')  # Write the data to a CSV file
        return None
    print("Daily returns calculated.")
    return data

data = get_stock_data("AAPL", "2015-01-01", "2025-01-01")

if data is not None:
    # Step 2: Feature Engineering
    def add_technical_indicators(data, ticker):
        print("Adding technical indicators...")
        data['SMA_10'] = data[('Close', ticker)].rolling(window=10).mean()
        data['SMA_50'] = data[('Close', ticker)].rolling(window=50).mean()
        data['RSI'] = 100 - (100 / (1 + (data['Return'].rolling(window=14).mean() / data['Return'].rolling(window=14).std())))
        data['MACD'] = data[('Close', ticker)].ewm(span=12, adjust=False).mean() - data[('Close', ticker)].ewm(span=26, adjust=False).mean()
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data = data.dropna()
        print("Technical indicators added.")
        return data

    data = add_technical_indicators(data, "AAPL")

    print("Creating target labels...")
    data['Target'] = np.where(data[('Close', "AAPL")].shift(-1) > data[('Close', "AAPL")], 1, 0)
    print("Target labels created.")
    
    # Additional debug prints to ensure the script runs completely
    print("Data after feature engineering and target label creation:")
    print(data.head())
    
    # Write the final data to a CSV file
    data.to_csv('processed_data.csv')
else:
    print("Data collection failed. Please check the column names and try again.")