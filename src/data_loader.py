# src/data_loader.py [UPDATED]

import pandas as pd
import yfinance as yf
import requests
from io import StringIO
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable for the API key (consider setting it in a secure way for production)
ALPHA_VANTAGE_API_KEY = '8E08KRXT8U5EBW3Z'

def fetch_alpha_vantage_data(symbol="GC=F", output_size="full"):
    """
    Fetches daily gold trading data from Alpha Vantage.

    Args:
        symbol (str): The ticker symbol for the commodity. Default is 'GC=F' (Gold Futures).
        output_size (str): Amount of data to retrieve ('compact' or 'full'). Default is 'full'.

    Returns:
        pd.DataFrame: DataFrame containing the fetched data with 'Close' prices indexed by timestamp.
    """
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize={output_size}&apikey={ALPHA_VANTAGE_API_KEY}&datatype=csv'
    response = requests.get(url)
    
    # Check for successful response and parse data
    if response.status_code == 200 and "timestamp" in response.text:
        try:
            data = pd.read_csv(StringIO(response.text), parse_dates=["timestamp"])
            data.set_index("timestamp", inplace=True)
            logger.info("Alpha Vantage data fetched successfully.")
            return data[['close']].rename(columns={"close": "Close"})
        except pd.errors.ParserError as e:
            logger.error(f"Failed to parse Alpha Vantage data: {e}")
            return pd.DataFrame()
    else:
        logger.warning("Alpha Vantage data is empty or API limit reached.")
        return pd.DataFrame()

def fetch_yahoo_finance_data(symbol="GC=F", start_date="2010-01-01"):
    """
    Fetches daily gold trading data from Yahoo Finance.

    Args:
        symbol (str): The ticker symbol for the commodity. Default is 'GC=F' (Gold Futures).
        start_date (str): The start date for fetching historical data. Default is '2010-01-01'.

    Returns:
        pd.DataFrame: DataFrame containing the 'Close' prices indexed by date.
    """
    try:
        yahoo_data = yf.download(symbol, interval='1d', start=start_date)[['Close']]
        logger.info("Yahoo Finance data fetched successfully.")
        return yahoo_data
    except Exception as e:
        logger.error(f"Failed to fetch Yahoo Finance data: {e}")
        return pd.DataFrame()

def get_combined_data(symbol="GC=F", start_date="2010-01-01"):
    """
    Fetches and combines gold trading data from Alpha Vantage and Yahoo Finance.

    Args:
        symbol (str): The ticker symbol for the commodity. Default is 'GC=F' (Gold Futures).
        start_date (str): The start date for fetching historical data.

    Returns:
        pd.DataFrame: DataFrame with combined 'Close' prices indexed by date. Returns an empty DataFrame if both sources fail.
    """
    alpha_data = fetch_alpha_vantage_data(symbol)
    yahoo_data = fetch_yahoo_finance_data(symbol, start_date)
    
    # Combine data sources, prioritizing Alpha Vantage, then Yahoo Finance if available
    if not alpha_data.empty and not yahoo_data.empty:
        logger.info("Combining Alpha Vantage and Yahoo Finance data.")
        combined_data = alpha_data.join(yahoo_data, rsuffix='_yahoo', how='outer').dropna()
    elif not alpha_data.empty:
        logger.info("Only Alpha Vantage data available; using it exclusively.")
        combined_data = alpha_data
    elif not yahoo_data.empty:
        logger.info("Only Yahoo Finance data available; using it exclusively.")
        combined_data = yahoo_data
    else:
        logger.warning("No data available from Alpha Vantage or Yahoo Finance.")
        combined_data = pd.DataFrame()

    return combined_data
