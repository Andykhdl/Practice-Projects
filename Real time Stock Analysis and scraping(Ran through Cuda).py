import requests 
import pandas as pd
import time
import matplotlib.pyplot as plt 
import seaborn as sns
# import cupy as cp 

API_KEY = "your_alpha_vantage_api_key"
BASE_URL = "https://www.alphavantage.co/query"

def fetch_stock_data(symbol):
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": "1min",
        "apikey": API_KEY,
        "outputsize": "full"
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    print(data)
#    if "Time Series (1min)" in data:
#        df = pd.DataFrame.from_dict(data["Time Series (1min)"], orient='index').astyper(float)
#        df.columns = ["Open","High","Low","Close","Volume"]
#        df.index = pd.to_datetime(df.index)
#    else:
#        print("Error fetching data:", data.get("Note", "Unknown error"))
#        return None
#
## Example: Fetch data for AAPL
stock_data = fetch_stock_data("AAPL")
#stock_data.head()
##stock_data.info()
#stock_data.describe()
#
#for column in stock_data.columns:
#    plt.figure(figsize=(8, 5))
#    plt.plot(stock_data[column])
#    plt.title(f'{column}')
#    plt.xlabel('Time')
#    plt.ylabel(column)
#    plt.show()
#
## Convert to NumPy-like structure for correlation
## stock_data_gpu = cp.array(stock_data.to_pandas().values)  # Convert CuDF to NumPy-like array
## correlation_matrix = cp.corrcoef(stock_data_gpu, rowvar=False)
## correlation_matrix = cp.asnumpy(correlation_matrix)  # Convert back to NumPy
#
## Plot correlation heatmap
#plt.figure(figsize=(15, 10))
#sns.heatmap(correlation_matrix, annot=True, annot_kws={"size": 12})
#plt.show()