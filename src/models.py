# src/models.py

import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing



def run_lstm(scaled_data, scaler, callback=None, epochs=2, batch_size=1, units=50, learning_rate=0.001, lookback=30):
    """Trains an LSTM model with custom parameters and returns the forecast and MSE. Updates the GUI through a callback."""
    generator = TimeseriesGenerator(scaled_data, scaled_data, length=lookback, batch_size=batch_size)
    
    model = Sequential([
        LSTM(units, activation='relu', input_shape=(lookback, scaled_data.shape[1])),
        Dense(scaled_data.shape[1])
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    
    # Train the model, logging progress at each epoch
    for epoch in range(epochs):
        history = model.fit(generator, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        # Update the GUI with the progress, if callback is provided
        if callback:
            callback(f"Epoch {epoch+1}/{epochs}, loss: {loss}", epoch+1, epochs)
    
    # Generate predictions
    forecast = []
    input_data = scaled_data[-lookback:]
    for _ in range(5):
        prediction = model.predict(input_data.reshape((1, lookback, scaled_data.shape[1])))
        forecast.append(prediction[0, 0])
        input_data = np.vstack([input_data[1:], prediction])

    # Calculate MSE for the forecast
    forecast_reshaped = np.array(forecast).reshape(-1, 1)
    forecast_inverse = scaler.inverse_transform(np.hstack([forecast_reshaped, np.zeros((5, scaled_data.shape[1] - 1))]))[:, 0]
    mse = mean_squared_error(scaled_data[-5:, 0], forecast)
    
    return forecast_inverse, mse





def run_ets(data):
    model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=365)
    model_fit = model.fit()
    forecast = model_fit.forecast(5)
    return forecast, mean_squared_error(data[-5:], forecast)




def run_arima(data):
    # Set frequency if missing
    if data.index.freq is None:
        data = data.asfreq('D')  # Assumes daily data; adjust as necessary
    
    # Drop NaN values in data
    data = data.dropna()

    print("\n================ Running ARIMA Model ================\nStarting ARIMA model training...")
    try:
        # Suppress all warnings temporarily
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore all warnings
            
            model = ARIMA(data, order=(5,1,0))
            model_fit = model.fit()
            forecast = model_fit.forecast(5)
        
        # Calculate MSE between the last 5 actual data points and the forecast
        mse = mean_squared_error(data[-5:], forecast)
        forecast = forecast.rename("Forecast")
        
        print("[INFO] ARIMA model training completed.\n")
        return forecast, mse
    except Exception as e:
        print(f"ARIMA Model failed: {e}")
        return None, float('inf')






def run_linear_regression(data):
    X = np.arange(len(data)).reshape(-1, 1)
    y = data['Close'].values
    model = LinearRegression()
    model.fit(X, y)
    future_X = np.arange(len(data), len(data) + 5).reshape(-1, 1)
    forecast = model.predict(future_X)
    return forecast, mean_squared_error(y[-5:], forecast)


