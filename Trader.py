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
    if ('Close', ticker) in data.columns:
        data['Return'] = data[('Close', ticker)].pct_change()
    else:
        print(f"Column 'Close' for ticker {ticker} not found. Available columns:", data.columns)
        return None
    print("Daily returns calculated.")
    return data

data = get_stock_data("NVDA", "2023-01-01", "2025-01-01")

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

    data = add_technical_indicators(data, "NVDA")

    print("Creating target labels...")
    data['Target'] = np.where(data[('Close', "NVDA")].shift(-1) > data[('Close', "NVDA")], 1, 0)
    print("Target labels created.")

    features = ['SMA_10', 'SMA_50', 'RSI', 'MACD', 'Signal_Line']
    X = data[features]
    y = data['Target']
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    print(f"Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")

    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10]
    }

    print("Performing Grid Search for best hyperparameters...")
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Grid Search completed.")

    print("Extracting best model...")
    best_model = grid_search.best_estimator_
    print("Making predictions on the test set...")
    y_pred = best_model.predict(X_test)

    print("Best Parameters:", grid_search.best_params_)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("AUC-ROC:", roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]))

    print("Calculating feature importance...")
    feature_importances = pd.DataFrame({
        'Feature': features,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print(feature_importances)

    print("Visualizing feature importance...")
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importances['Feature'], feature_importances['Importance'])
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.show()

    print("Comparing predictions vs actual trends...")
    plt.figure(figsize=(14, 7))
    actual_prices = data[('Close', 'NVDA')][-len(y_test):]
    predicted_uptrend = pd.Series(y_pred, index=actual_prices.index)

    plt.plot(actual_prices, label='Actual Prices', alpha=0.8)
    plt.plot(actual_prices.index, data['SMA_10'][-len(y_test):].values, label='SMA_10', alpha=0.6)
    plt.scatter(predicted_uptrend[predicted_uptrend == 1].index, actual_prices[predicted_uptrend == 1], 
                label='Predicted Uptrend', color='green', alpha=0.7)
    plt.legend()
    plt.show()
else:
    print("Data collection failed. Please check the column names and try again.")