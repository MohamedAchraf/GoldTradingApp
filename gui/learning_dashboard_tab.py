# gui/learning_dashboard_tab.py

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QRadioButton, QCheckBox,
    QPushButton, QTextEdit, QProgressBar, QApplication
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import yfinance as yf
from datetime import datetime, timedelta
from train_worker import TrainWorker
from src.preprocessing import preprocess_data
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd


class LearningDashboardTab(QWidget):
    """Tab for model selection, data sources, and running predictions."""

    def __init__(self, arima_tab, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.lstm_tab = None  # Placeholder for LSTM tab instance
        self.arima_tab = arima_tab  # Reference to ARIMA tab for order retrieval
        self.data = None  # Placeholder for fetched data
        self.init_ui()

    def set_lstm_tab(self, lstm_tab):
        """Sets the LSTM tab to access its parameters."""
        self.lstm_tab = lstm_tab

    def init_ui(self):
        """Initializes the UI components."""
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()

        # Model Selection
        model_group = QGroupBox("Models")
        model_layout = QVBoxLayout()
        self.model_buttons = {
            "LSTM": QRadioButton("LSTM"),
            "ARIMA": QRadioButton("ARIMA"),
            "LR": QRadioButton("Linear Regression"),
            "ETS": QRadioButton("ETS")
        }
        for button in self.model_buttons.values():
            model_layout.addWidget(button)
        model_group.setLayout(model_layout)

        # Data Sources
        data_group = QGroupBox("Data Sources")
        data_layout = QVBoxLayout()
        self.data_sources = {
            "Yahoo Finance": QCheckBox("Yahoo Finance"),
            "Alpha Vantage": QCheckBox("Alpha Vantage"),
            "Local Data": QCheckBox("Local Data")
        }
        for checkbox in self.data_sources.values():
            data_layout.addWidget(checkbox)
        data_group.setLayout(data_layout)

        top_layout.addWidget(data_group)
        top_layout.addWidget(model_group)

        # Console and Buttons
        button_layout = QHBoxLayout()
        self.display_data_button = QPushButton("Fetch Data")
        self.display_data_button.setIcon(QIcon("assets/icons/cloud-download.svg"))
        self.display_data_button.clicked.connect(self.display_data)

        self.start_prediction_button = QPushButton("Start Prediction")
        self.start_prediction_button.setIcon(QIcon("assets/icons/play.svg"))
        self.start_prediction_button.clicked.connect(self.start_prediction)

        button_layout.addWidget(self.display_data_button)
        button_layout.addWidget(self.start_prediction_button)

        # Output Console
        self.output_console = QTextEdit()
        self.output_console.setReadOnly(True)
        self.output_console.setStyleSheet("background-color: black; color: lightgreen; font-family: Courier; font-size: 10pt;")
        
        # Additional Buttons: Copy and Clear (Placed below the output console)
        console_button_layout = QHBoxLayout()
        self.copy_button = QPushButton("Copy")
        self.copy_button.setIcon(QIcon("assets/icons/copy.svg"))
        self.copy_button.clicked.connect(self.copy_output)

        self.clear_button = QPushButton("Clear")
        self.clear_button.setIcon(QIcon("assets/icons/trash.svg"))
        self.clear_button.clicked.connect(self.clear_output)

        console_button_layout.addWidget(self.copy_button)
        console_button_layout.addWidget(self.clear_button)

        # Progress Bar
        self.progress_bar = QProgressBar()

        # Add layouts and widgets to the main layout
        main_layout.addLayout(top_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(QLabel("Output:"))
        main_layout.addWidget(self.output_console)
        main_layout.addLayout(console_button_layout)  # Add Copy and Clear buttons under the console
        main_layout.addWidget(self.progress_bar)
        self.setLayout(main_layout)

    def display_data(self):
        """Fetches and displays sample data."""
        if self.data_sources["Yahoo Finance"].isChecked():
            self.output_console.append("Fetching data from Yahoo Finance...")
            try:
                self.data = yf.download('GC=F', interval='1d', start='2010-01-01')[['Close']]
                if not self.data.empty:
                    self.output_console.append("Data fetched successfully. Displaying last 5 rows:\n")
                    self.output_console.append(str(self.data.tail()))
                    # Set the fetched data in ARIMA tab
                    self.arima_tab.set_data(self.data)  # Ensure this calls the correct ARIMATab method
                else:
                    self.output_console.append("Failed to fetch data or data is empty.")
            except Exception as e:
                self.output_console.append(f"Error fetching data: {e}")
        else:
            self.output_console.append("Please select Yahoo Finance as the data source to display data.")

    def start_prediction(self):
        """Starts the prediction process based on selected model and parameters."""
        selected_model = [key for key, button in self.model_buttons.items() if button.isChecked()]
        if not selected_model:
            self.output_console.append("No model selected. Please select a model.")
            return
        elif not hasattr(self, 'data') or self.data is None:
            self.output_console.append("No data available. Please fetch data first.")
            return

        model_name = selected_model[0]
        self.output_console.append(f"Starting prediction using {model_name} model...")

        # Preprocess data
        combined_data = self.data.dropna()
        scaled_data, scaler = preprocess_data(combined_data)

        if model_name == "ARIMA":
            # Retrieve order values from ARIMA tab
            p, d, q = self.arima_tab.get_order()
            self.output_console.append(f"Using ARIMA order: (p,d,q)=({p},{d},{q})")
            # Run ARIMA prediction with parameters
            self.run_arima_prediction(combined_data['Close'], p, d, q)  # Pass the 'Close' column

        elif model_name == "LSTM":
            if self.lstm_tab:
                # Fetch parameters from LSTM tab
                epochs = self.lstm_tab.epochs_input.value()
                batch_size = int(self.lstm_tab.batch_size_input.currentText())
                units = self.lstm_tab.units_slider.value()
                learning_rate = self.lstm_tab.learning_rate_input.value()
                lookback = self.lstm_tab.lookback_slider.value()

                # Run LSTM prediction with parameters
                self.train_lstm(scaled_data, scaler, epochs, batch_size, units, learning_rate, lookback)
            else:
                self.output_console.append("LSTM parameters not found.")
                return

    def run_arima_prediction(self, data, p, d, q):
        """Runs ARIMA prediction and displays results."""
        try:
            model = ARIMA(data, order=(p, d, q))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=5)  # Predict next 5 values
            mse = mean_squared_error(data[-5:], forecast)  # Calculate MSE on last 5 actual values

            self.output_console.append("Predicted 5-Day ARIMA Forecast:")
            for i, value in enumerate(forecast):
                self.output_console.append(f"Day {i + 1}: {value:.2f}")
            self.output_console.append(f"\nARIMA MSE: {mse:.4f}\n")

        except Exception as e:
            self.output_console.append(f"Error during ARIMA prediction: {e}")

    def train_lstm(self, scaled_data, scaler, epochs, batch_size, units, learning_rate, lookback):
        """Trains the LSTM model and updates progress."""
        self.worker = TrainWorker(
            scaled_data, scaler, epochs=epochs, batch_size=batch_size, 
            units=units, learning_rate=learning_rate, lookback=lookback
        )
        self.worker.progress.connect(self.output_console.append)
        self.worker.progress_value.connect(self.progress_bar.setValue)
        self.worker.result.connect(self.display_result)
        self.progress_bar.setValue(0)
        self.worker.start()

    def display_result(self, forecast, mse):
        """Displays forecast results and MSE."""
        start_date = datetime.now() + timedelta(days=1)
        prediction_dates = self.get_next_business_days(start_date, num_days=len(forecast))
        self.output_console.append("Predicted 5-Day Forecast:")
        for date, value in zip(prediction_dates, forecast):
            self.output_console.append(f"{date.strftime('%Y-%m-%d')}: {value:.2f}")
        if mse is not None:
            self.output_console.append(f"\nMSE: {mse:.4f}\n")
        self.progress_bar.setValue(100)

    def get_next_business_days(self, start_date, num_days=5):
        """Generate the next business days, skipping weekends."""
        business_days = []
        current_date = start_date
        while len(business_days) < num_days:
            if current_date.weekday() < 5:  # Monday=0, Sunday=6
                business_days.append(current_date)
            current_date += timedelta(days=1)
        return business_days

    def copy_output(self):
        """Copies the output console text to the clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.output_console.toPlainText())

    def clear_output(self):
        """Clears the output console text."""
        self.output_console.clear()
