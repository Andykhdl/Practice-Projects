import requests
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

API_KEY = "your_alpha_vantage_api_key"
BASE_URL = "https://www.alphavantage.co/query"

def fetch_stock_data(symbol):
    """
    Fetch intraday stock data for the given symbol from Alpha Vantage.
    """
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": "1min",
        "apikey": API_KEY,
        "outputsize": "full"
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    if "Time Series (1min)" in data:
        df = pd.DataFrame.from_dict(data["Time Series (1min)"], orient='index')
        df.columns = ["open", "high", "low", "close", "volume"]  # Rename columns for clarity
        df = df.astype(float)  # Convert all columns to float
        df.index = pd.to_datetime(df.index)  # Convert index to datetime
        return df.sort_index()  # Sort by time
    else:
        print("Error fetching data:", data.get("Note", "Unknown error"))
        return None

# Fetch data for AAPL
stock_data = fetch_stock_data("AAPL")
if stock_data is not None:
    print(stock_data.head())
    print(stock_data.describe())

    # Plot individual columns
    for column in stock_data.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(stock_data[column])
        plt.title(f'{column} Over Time')
        plt.xlabel('Time')
        plt.ylabel(column.capitalize())
        plt.grid(True)
        plt.show()

    # Calculate and plot the correlation matrix
    correlation_matrix = stock_data.corr()  # Use pandas correlation
    plt.figure(figsize=(10, 8))
#    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", annot_kws={"size": 12})
#    plt.title("Correlation Matrix")
#    plt.show()
else:
    print("No data to process.")
