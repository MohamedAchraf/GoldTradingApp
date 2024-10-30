# src/preprocessing.py

from sklearn.preprocessing import MinMaxScaler
import numpy as np

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler
