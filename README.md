# Stock Trend Prediction Using Machine Learning

This project demonstrates a machine learning pipeline to predict stock trends based on historical price data and technical indicators. The focus is on creating a classification model to identify whether the stock price will go up or down on the next trading day.

## Features

- **Data Collection**: Fetches historical stock data using Yahoo Finance.
- **Feature Engineering**: Computes key technical indicators such as:
  - Simple Moving Averages (SMA)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD) and Signal Line
- **Model Building**: Trains a **Random Forest Classifier** to predict stock trends using:
  - Hyperparameter tuning via Grid Search.
  - Metrics evaluation including Accuracy and AUC-ROC.
- **Visualization**:
  - Feature Importance.
  - SMA and actual stock prices.
  - Predicted uptrends plotted against actual trends.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/stock-trend-prediction.git
   cd stock-trend-prediction
   ```
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Update the `get_stock_data` function call with your desired stock ticker and date range:
   ```python
   data = get_stock_data("AAPL", "2015-01-01", "2023-01-01")
   ```
2. Run the script:
   ```bash
   python Trader.py
   ```

## Key Files

- `Trader.py`: Main script that includes data fetching, feature engineering, model training, and visualization.

## Dependencies

- `pandas`
- `numpy`
- `scikit-learn`
- `yfinance`
- `matplotlib`

Install dependencies with:
```bash
pip install pandas numpy scikit-learn yfinance matplotlib
```

## Output

1. **Console Output**:
   - Best hyperparameters identified during Grid Search.
   - Classification Report and Accuracy.
   - AUC-ROC Score.

2. **Visualizations**:
   - Bar chart of feature importance.
   - Actual vs Predicted stock trends.

## Limitations

- The model's predictions are based on historical data and technical indicators. They do not guarantee future performance.
- The dataset's quality and the chosen date range significantly affect the model's accuracy.

## Contributing

Feel free to fork the repository and submit pull requests for enhancements or bug fixes. Contributions are welcome!

## License

This project is licensed under the MIT License. See the `LICENSE` file for details. 
