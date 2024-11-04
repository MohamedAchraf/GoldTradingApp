# src/models.py

import warnings
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_lstm(scaled_data, scaler, callback=None, epochs=2, batch_size=1, units=50, learning_rate=0.001, lookback=30):
    """
    Trains an LSTM model with specified parameters and returns the forecast and Mean Squared Error (MSE).

    Args:
        scaled_data (np.ndarray): Scaled data used for training.
        scaler (Scaler): Scaler instance used to inverse-transform predictions.
        callback (function, optional): Function for updating GUI or console with training progress.
        epochs (int): Number of training epochs. Default is 2.
        batch_size (int): Batch size for training. Default is 1.
        units (int): Number of LSTM units. Default is 50.
        learning_rate (float): Learning rate for optimizer. Default is 0.001.
        lookback (int): Number of past time steps to use for prediction. Default is 30.

    Returns:
        np.ndarray: Inverse-transformed forecast values.
        float: Mean Squared Error (MSE) of the forecast.
    """
    try:
        generator = TimeseriesGenerator(scaled_data, scaled_data, length=lookback, batch_size=batch_size)
        model = Sequential([
            LSTM(units, activation='relu', input_shape=(lookback, scaled_data.shape[1])),
            Dense(scaled_data.shape[1])
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

        for epoch in range(epochs):
            history = model.fit(generator, epochs=1, verbose=0)
            loss = history.history['loss'][0]
            if callback:
                callback(f"Epoch {epoch+1}/{epochs}, loss: {loss}", epoch+1, epochs)

        forecast = []
        input_data = scaled_data[-lookback:]
        for _ in range(5):
            prediction = model.predict(input_data.reshape((1, lookback, scaled_data.shape[1])))
            forecast.append(prediction[0, 0])
            input_data = np.vstack([input_data[1:], prediction])

        forecast_reshaped = np.array(forecast).reshape(-1, 1)
        forecast_inverse = scaler.inverse_transform(np.hstack([forecast_reshaped, np.zeros((5, scaled_data.shape[1] - 1))]))[:, 0]
        mse = mean_squared_error(scaled_data[-5:, 0], forecast)
        
        logger.info("LSTM model training completed.")
        return forecast_inverse, mse
    except Exception as e:
        logger.error(f"LSTM Model failed: {e}")
        return None, float('inf')

def run_ets(data, trend='add', seasonal='add', seasonal_periods=365):
    """
    Trains an Exponential Smoothing (ETS) model and returns the forecast and MSE.

    Args:
        data (pd.Series): Time series data.
        trend (str): Type of trend component. Default is 'add'.
        seasonal (str): Type of seasonal component. Default is 'add'.
        seasonal_periods (int): Period of seasonality. Default is 365.

    Returns:
        pd.Series: Forecasted values.
        float: Mean Squared Error (MSE) of the forecast.
    """
    try:
        model = ExponentialSmoothing(data, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
        model_fit = model.fit()
        forecast = model_fit.forecast(5)
        mse = mean_squared_error(data[-5:], forecast)
        logger.info("ETS model training completed.")
        return forecast, mse
    except Exception as e:
        logger.error(f"ETS Model failed: {e}")
        return None, float('inf')

def run_arima(data, order=(5,1,0)):
    """
    Trains an ARIMA model and returns the forecast and MSE.

    Args:
        data (pd.Series): Time series data.
        order (tuple): Order of the ARIMA model. Default is (5,1,0).

    Returns:
        pd.Series: Forecasted values.
        float: Mean Squared Error (MSE) of the forecast.
    """
    if data.index.freq is None:
        data = data.asfreq('D')
    
    data = data.dropna()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(data, order=order)
            model_fit = model.fit()
            forecast = model_fit.forecast(5)
        
        mse = mean_squared_error(data[-5:], forecast)
        logger.info("ARIMA model training completed.")
        return forecast.rename("Forecast"), mse
    except Exception as e:
        logger.error(f"ARIMA Model failed: {e}")
        return None, float('inf')

def run_linear_regression(data):
    """
    Trains a Linear Regression model and returns the forecast and MSE.

    Args:
        data (pd.DataFrame): DataFrame with 'Close' prices.

    Returns:
        np.ndarray: Forecasted values.
        float: Mean Squared Error (MSE) of the forecast.
    """
    try:
        X = np.arange(len(data)).reshape(-1, 1)
        y = data['Close'].values
        model = LinearRegression()
        model.fit(X, y)
        future_X = np.arange(len(data), len(data) + 5).reshape(-1, 1)
        forecast = model.predict(future_X)
        mse = mean_squared_error(y[-5:], forecast)
        logger.info("Linear Regression model training completed.")
        return forecast, mse
    except Exception as e:
        logger.error(f"Linear Regression Model failed: {e}")
        return None, float('inf')
