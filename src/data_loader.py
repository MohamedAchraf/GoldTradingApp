# src/data_loader.py

import pandas as pd
import yfinance as yf
import requests
from io import StringIO

ALPHA_VANTAGE_API_KEY = '8E08KRXT8U5EBW3Z'

def fetch_alpha_vantage_data():
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=GC=F&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}&datatype=csv'
    response = requests.get(url)
    if response.status_code == 200 and "timestamp" in response.text:
        try:
            data = pd.read_csv(StringIO(response.text), parse_dates=["timestamp"])
            data.set_index("timestamp", inplace=True)
            return data[['close']].rename(columns={"close": "Close"})
        except Exception as e:
            print(f"Failed to parse Alpha Vantage data: {e}")
            return pd.DataFrame()
    else:
        print("Alpha Vantage data is empty or API limit reached.")
        return pd.DataFrame()

def fetch_yahoo_finance_data():
    yahoo_data = yf.download('GC=F', interval='1d', start='2010-01-01')[['Close']]
    return yahoo_data

def get_combined_data():
    alpha_data = fetch_alpha_vantage_data()
    yahoo_data = fetch_yahoo_finance_data()
    if not alpha_data.empty and not yahoo_data.empty:
        combined_data = alpha_data.join(yahoo_data, rsuffix='_yahoo', how='outer').dropna()
    elif not alpha_data.empty:
        combined_data = alpha_data
    elif not yahoo_data.empty:
        combined_data = yahoo_data
    else:
        combined_data = pd.DataFrame()
    return combined_data
