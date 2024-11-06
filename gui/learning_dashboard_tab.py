# gui/learning_dashboard_tab.py

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QRadioButton, QCheckBox, QPushButton, QTextEdit, QLabel, QProgressBar, QHBoxLayout
)
from PyQt5.QtGui import QIcon
import yfinance as yf
from src.preprocessing import preprocess_data
from src.models import run_lstm, run_arima, run_linear_regression, run_ets


class LearningDashboardTab(QWidget):
    # (Class code remains the same)

    """Defines the Learning Dashboard tab."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None  # Placeholder for fetched data
        self.init_ui()

    def init_ui(self):
        """Initializes the Learning Dashboard tab UI."""
        layout = QVBoxLayout()
        
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
            "Yahoo Finance": QCheckBox("YAHOO Finance"),
            "Alpha Vantage": QCheckBox("Alpha Vantage"),
            "Local Data": QCheckBox("Local Data")
        }
        for checkbox in self.data_sources.values():
            data_layout.addWidget(checkbox)
        data_group.setLayout(data_layout)
        
        # Buttons and Console Output
        button_layout = QHBoxLayout()
        self.display_data_button = QPushButton("Fetch Data")
        self.start_prediction_button = QPushButton("Start Prediction")
        self.display_data_button.setIcon(QIcon("assets/icons/database-down.svg"))
        self.start_prediction_button.setIcon(QIcon("assets/icons/play.svg"))
        button_layout.addWidget(self.start_prediction_button)
        button_layout.addWidget(self.display_data_button)
        
        # Console Output and Progress Bar
        self.output_console = create_console_output()
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        # Arrange layout
        top_layout = QHBoxLayout()
        top_layout.addWidget(model_group)
        top_layout.addWidget(data_group)
        layout.addLayout(top_layout)
        layout.addLayout(button_layout)
        layout.addWidget(QLabel("Output:"))
        layout.addWidget(self.output_console)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

        # Connect buttons to their respective functions
        self.display_data_button.clicked.connect(self.fetch_data)
        self.start_prediction_button.clicked.connect(self.start_prediction)

    def fetch_data(self):
        """Fetches and displays a sample of data from Yahoo Finance."""
        if self.data_sources["Yahoo Finance"].isChecked():
            self.output_console.append("Fetching data from Yahoo Finance...")
            self.data = yf.download('GC=F', interval='1d', start='2010-01-01')[['Close']]
            if not self.data.empty:
                self.output_console.append("Data fetched successfully. Displaying last 5 rows:\n")
                self.output_console.append(str(self.data.tail()))
            else:
                self.output_console.append("Failed to fetch data or data is empty.")
        else:
            self.output_console.append("Please select Yahoo Finance as the data source to display data.")

    def start_prediction(self):
        """Starts the prediction process using the selected model and fetched data."""
        selected_model = [key for key, button in self.model_buttons.items() if button.isChecked()]
        
        if not selected_model:
            self.output_console.append("No model selected. Please select a model.")
            return
        elif self.data is None:
            self.output_console.append("No data available. Please fetch data first.")
            return

        model_name = selected_model[0]
        self.output_console.append(f"Starting prediction using {model_name} model...")

        # Preprocess data if needed
        combined_data = self.data.dropna()
        scaled_data, scaler = preprocess_data(combined_data)

        # Determine which model to use and execute the prediction
        if model_name == "LSTM":
            # Placeholder for running LSTM model, typically with additional parameters
            forecast, mse = run_lstm(scaled_data, scaler, epochs=2, batch_size=1, units=50, learning_rate=0.001, lookback=30)
            self.display_result(forecast, mse)
        elif model_name == "ARIMA":
            forecast, mse = run_arima(combined_data['Close'])
            self.display_result(forecast, mse)
        elif model_name == "LR":
            forecast, mse = run_linear_regression(combined_data)
            self.display_result(forecast, mse)
        elif model_name == "ETS":
            forecast, mse = run_ets(combined_data['Close'])
            self.display_result(forecast, mse)
        else:
            self.output_console.append("Unknown model selected.")

    def display_result(self, forecast, mse):
        """Displays the final forecast with dates."""
        self.output_console.append("Predicted 5-Day Forecast:")
        for value in forecast:
            self.output_console.append(f"{value:.2f}")
        
        # Optionally display MSE
        if mse is not None:
            self.output_console.append(f"\nMSE: {mse:.4f}\n")
        self.progress_bar.setValue(100)
