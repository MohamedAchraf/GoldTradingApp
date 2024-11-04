# gui/train_worker.py

import sys
import os

# Add the root of the project to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use absolute import
from src.models import run_lstm


from PyQt5.QtCore import QThread, pyqtSignal

class TrainWorker(QThread):
    """Worker thread for training the LSTM model to keep the GUI responsive."""
    progress = pyqtSignal(str)
    result = pyqtSignal(object, float)
    progress_value = pyqtSignal(int)

    def __init__(self, scaled_data, scaler, epochs=2, batch_size=1, units=50, learning_rate=0.001, lookback=30):
        super().__init__()
        self.scaled_data = scaled_data
        self.scaler = scaler
        self.epochs = epochs
        self.batch_size = batch_size
        self.units = units
        self.learning_rate = learning_rate
        self.lookback = lookback

    def run(self):
        def callback(message, epoch, total_epochs):
            self.progress.emit(message)
            self.progress_value.emit(int((epoch / total_epochs) * 100))

        forecast, mse = run_lstm(
            self.scaled_data, self.scaler, callback=callback,
            epochs=self.epochs, batch_size=self.batch_size, units=self.units,
            learning_rate=self.learning_rate, lookback=self.lookback
        )
        self.result.emit(forecast, mse)
        self.progress_value.emit(100)
