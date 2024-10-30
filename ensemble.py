#file scripts/TMP/gold_price_forecasting_ensemble.py

import warnings
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from io import StringIO

warnings.filterwarnings("ignore")

# Alpha Vantage API key (replace with your actual key)
ALPHA_VANTAGE_API_KEY = '8E08KRXT8U5EBW3Z'

def fetch_alpha_vantage_data():
    print("[INFO] Fetching data from Alpha Vantage...")
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=GC=F&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}&datatype=csv'
    response = requests.get(url)
    if response.status_code == 200 and "timestamp" in response.text:
        try:
            data = pd.read_csv(StringIO(response.text), parse_dates=["timestamp"])
            data.set_index("timestamp", inplace=True)
            print("[INFO] Alpha Vantage Data Fetched Successfully.\n")
            return data[['close']].rename(columns={"close": "Close"})
        except Exception as e:
            print(f"[ERROR] Failed to parse Alpha Vantage data: {e}")
            return pd.DataFrame()
    else:
        print("[ERROR] Alpha Vantage data is empty or API limit reached.")
        return pd.DataFrame()

def fetch_yahoo_finance_data():
    print("[INFO] Fetching data from Yahoo Finance...")
    yahoo_data = yf.download('GC=F', interval='1d', start='2010-01-01')[['Close']]
    print("[INFO] Yahoo Finance Data Fetched Successfully.\n")
    return yahoo_data

def get_combined_data():
    print("[INFO] Fetching and combining data from Alpha Vantage and Yahoo Finance...")
    alpha_data = fetch_alpha_vantage_data()
    yahoo_data = fetch_yahoo_finance_data()
    
    if not alpha_data.empty and not yahoo_data.empty:
        print("[INFO] Combining Alpha Vantage and Yahoo Finance data...")
        combined_data = alpha_data.join(yahoo_data, rsuffix='_yahoo', how='outer').dropna()
        print("[INFO] Data combined successfully.\n")
    elif not alpha_data.empty:
        print("[WARNING] Only Alpha Vantage data is available. Proceeding with Alpha Vantage data.\n")
        combined_data = alpha_data
    elif not yahoo_data.empty:
        print("[WARNING] Only Yahoo Finance data is available. Proceeding with Yahoo Finance data.\n")
        combined_data = yahoo_data
    else:
        print("[ERROR] No data available from Alpha Vantage or Yahoo Finance.")
        combined_data = pd.DataFrame()
    
    return combined_data

def preprocess_data(data):
    print("[INFO] Preprocessing data for training...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    print("[INFO] Data preprocessed successfully.")
    return scaled_data, scaler

def generate_timeseries_data(scaled_data, lookback=30):
    generator = TimeseriesGenerator(scaled_data, scaled_data, length=lookback, batch_size=1)
    return generator

def run_ets(data):
    print("\n================ Running ETS Model ================\nStarting ETS model training...")
    try:
        model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=365)
        model_fit = model.fit()
        forecast = model_fit.forecast(5)
        print("[INFO] ETS model training completed.\n")
        return forecast, mean_squared_error(data[-5:], forecast)
    except Exception as e:
        print(f"ETS Model failed: {e}")
        return None, float('inf')

def run_arima(data):
    print("\n================ Running ARIMA Model ================\nStarting ARIMA model training...")
    try:
        model = ARIMA(data, order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(5)
        print("[INFO] ARIMA model training completed.\n")
        return forecast, mean_squared_error(data[-5:], forecast)
    except Exception as e:
        print(f"ARIMA Model failed: {e}")
        return None, float('inf')

def run_linear_regression(data):
    print("\n================ Running Linear Regression Model ================\nStarting Linear Regression training...")
    try:
        X = np.arange(len(data)).reshape(-1, 1)
        y = data['Close'].values
        model = LinearRegression()
        model.fit(X, y)
        future_X = np.arange(len(data), len(data) + 5).reshape(-1, 1)
        forecast = model.predict(future_X)
        print("[INFO] Linear Regression model training and forecasting completed.\n")
        return forecast, mean_squared_error(y[-5:], forecast)
    except Exception as e:
        print(f"Linear Regression failed: {e}")
        return None, float('inf')

def run_lstm(data, lookback=30, epochs=10):
    print("\n================ Running LSTM Model ================\nStarting LSTM model training...")
    try:
        generator = generate_timeseries_data(data, lookback=lookback)
        
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(lookback, data.shape[1])),
            Dense(data.shape[1])
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        model.fit(generator, epochs=epochs, verbose=1)
        
        forecast = []
        input_data = data[-lookback:]
        for _ in range(5):
            prediction = model.predict(input_data.reshape((1, lookback, data.shape[1])))
            forecast.append(prediction[0, 0])
            input_data = np.vstack([input_data[1:], prediction])
        
        print("[INFO] LSTM model training and forecasting completed.\n")
        return forecast, mean_squared_error(data[-5:, 0], forecast)
    except Exception as e:
        print(f"LSTM Model failed: {e}")
        return None, float('inf')

def main():
    # Load and preprocess the data
    combined_data = get_combined_data()
    scaled_data, scaler = preprocess_data(combined_data)
    
    # Run the models
    ets_forecast, ets_mse = run_ets(combined_data['Close'])
    arima_forecast, arima_mse = run_arima(combined_data['Close'])
    lr_forecast, lr_mse = run_linear_regression(combined_data)
    lstm_forecast, lstm_mse = run_lstm(scaled_data)
    
    # Convert LSTM forecast back to original scale
    lstm_forecast_reshaped = np.array(lstm_forecast).reshape(-1, 1)
    lstm_forecast_inverse = scaler.inverse_transform(np.hstack([lstm_forecast_reshaped, np.zeros((5, scaled_data.shape[1] - 1))]))[:, 0]
    
    # Ensure all forecasts are 1-dimensional
    ets_forecast = np.array(ets_forecast).flatten() if ets_forecast is not None else np.array([None]*5)
    arima_forecast = np.array(arima_forecast).flatten() if arima_forecast is not None else np.array([None]*5)
    lr_forecast = np.array(lr_forecast).flatten() if lr_forecast is not None else np.array([None]*5)
    lstm_forecast_inverse = np.array(lstm_forecast_inverse).flatten() if lstm_forecast_inverse is not None else np.array([None]*5)

    # Determine the best model
    mse_scores = {"ETS": ets_mse, "ARIMA": arima_mse, "Linear Regression": lr_mse, "LSTM": lstm_mse}
    best_model_name = min(mse_scores, key=mse_scores.get)
    best_forecast = {
        "ETS": ets_forecast,
        "ARIMA": arima_forecast,
        "Linear Regression": lr_forecast,
        "LSTM": lstm_forecast_inverse
    }[best_model_name]
    
    # Display results
    dates = pd.date_range(combined_data.index[-1], periods=5, freq='D')
    forecast_df = pd.DataFrame({
        'Date': dates,
        'ETS': ets_forecast,
        'ARIMA': arima_forecast,
        'Linear Regression': lr_forecast,
        'LSTM': lstm_forecast_inverse
    })
    print("\n================ Forecasts Table ================\n")
    print(forecast_df)
    print(f"\n[INFO] Best Model: {best_model_name}")
    print(f"[INFO] Forecast for next day by {best_model_name}: {best_forecast[0]}\n")

if __name__ == "__main__":
    main()
