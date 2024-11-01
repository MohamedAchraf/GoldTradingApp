# gui/main_gui.py

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
import sys
import os
import json
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

# Import necessary functions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models import run_lstm, run_arima, run_linear_regression, run_ets
from src.preprocessing import preprocess_data

class TrainWorker(QThread):
    """Worker thread for training the LSTM model to keep the GUI responsive."""
    progress = pyqtSignal(str)  # Signal to send progress updates to the GUI
    result = pyqtSignal(object, float)  # Signal to send the final result back to the GUI
    progress_value = pyqtSignal(int)  # Signal to update the progress bar in the GUI

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
        self.progress_value.emit(100)  # Complete progress

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Learning Dashboard")
        self.setGeometry(100, 100, 800, 600)

        # Main layout and tab setup
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        self.tabs = QTabWidget()
        
        # Add tabs
        self.tabs.addTab(self.create_dashboard_tab(), "Learning Dashboard")
        self.tabs.setTabIcon(0, QIcon("assets/icons/cpu.svg"))

        self.tabs.addTab(self.create_lstm_tab(), "LSTM")
        self.tabs.setTabIcon(1, QIcon("assets/icons/feather.svg"))

        self.tabs.addTab(self.create_model_tab("ARIMA"), "ARIMA")
        self.tabs.setTabIcon(2, QIcon("assets/icons/fingerprint.svg"))

        self.tabs.addTab(self.create_model_tab("LR"), "Linear Regression")
        self.tabs.setTabIcon(3, QIcon("assets/icons/flag.svg"))

        self.tabs.addTab(self.create_model_tab("ETS"), "ETS")
        self.tabs.setTabIcon(4, QIcon("assets/icons/flower1.svg"))

        
        main_layout.addWidget(self.tabs)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Connect buttons
        self.display_data_button.clicked.connect(self.display_data)
        self.display_data_button.setIcon(QIcon("assets/icons/database-down.svg"))

        self.start_prediction_button.clicked.connect(self.start_prediction)
        self.start_prediction_button.setIcon(QIcon("assets/icons/play.svg"))

        self.clear_output_button.clicked.connect(self.clear_output)
        self.clear_output_button.setIcon(QIcon("assets/icons/x-square-fill.svg"))

        self.copy_output_button.clicked.connect(self.copy_output)
        self.copy_output_button.setIcon(QIcon("assets/icons/copy.svg"))


        self.data = None  # Placeholder for data

    def create_model_tab(self, model_name):
        """Creates a model-specific tab with a placeholder for model controls."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Display a label with the model name
        layout.addWidget(QLabel(f"Parameter controls for {model_name} model"))

        # No specific parameters for ARIMA, Linear Regression, and ETS currently
        tab.setLayout(layout)
        return tab


    def create_dashboard_tab(self):
        """Main dashboard tab for model selection and data sources."""
        tab = QWidget()
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
        
        # Buttons
        button_layout = QHBoxLayout()
        self.display_data_button = QPushButton("Fetch Data")
        self.start_prediction_button = QPushButton("Start Prediction")
        button_layout.addWidget(self.start_prediction_button)
        button_layout.addWidget(self.display_data_button)
        
        # Console Output TextEdit (Unix-like style)
        self.output_console = QTextEdit()
        self.output_console.setReadOnly(True)
        self.output_console.setStyleSheet("background-color: black; color: lightgreen; font-family: Courier; font-size: 10pt;")

        # Output Controls
        output_control_layout = QHBoxLayout()
        self.clear_output_button = QPushButton("Clear")
        self.copy_output_button = QPushButton("Copy")
        output_control_layout.addWidget(self.clear_output_button)
        output_control_layout.addWidget(self.copy_output_button)
        
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        # Layout setup
        top_layout = QHBoxLayout()
        top_layout.addWidget(model_group)
        top_layout.addWidget(data_group)
        
        layout.addLayout(top_layout)
        layout.addLayout(button_layout)
        layout.addWidget(QLabel("Output:"))
        layout.addWidget(self.output_console)
        layout.addLayout(output_control_layout)
        layout.addWidget(self.progress_bar)
        
        tab.setLayout(layout)
        return tab

#from PyQt5.QtWidgets import (
#    QGroupBox, QLabel, QSpinBox, QSlider, QDoubleSpinBox, QVBoxLayout, QHBoxLayout, QWidget, QComboBox, QPushButton
#)
#from PyQt5.QtCore import Qt

    def create_lstm_tab(self):
        """Creates the LSTM model tab with elegant, organized UI components for parameter inputs."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Create parameter groups
        # Epochs Group
        epochs_group = QGroupBox("Epoch Settings")
        epochs_layout = QHBoxLayout()
        epochs_label = QLabel("Epochs:")
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 100)
        self.epochs_input.setValue(2)
        self.epochs_input.setToolTip("Number of complete passes through the training data.")
        epochs_layout.addWidget(epochs_label)
        epochs_layout.addWidget(self.epochs_input)
        epochs_group.setLayout(epochs_layout)
        layout.addWidget(epochs_group)

        # Batch Size Group
        batch_size_group = QGroupBox("Batch Size Settings")
        batch_size_layout = QHBoxLayout()
        batch_size_label = QLabel("Batch Size:")
        self.batch_size_input = QComboBox()
        self.batch_size_input.addItems(["1", "16", "32", "64", "128"])
        self.batch_size_input.setToolTip("Number of samples processed before updating the model.")
        batch_size_layout.addWidget(batch_size_label)
        batch_size_layout.addWidget(self.batch_size_input)
        batch_size_group.setLayout(batch_size_layout)
        layout.addWidget(batch_size_group)

        # Units Group
        units_group = QGroupBox("LSTM Units")
        units_layout = QHBoxLayout()
        units_label = QLabel("LSTM Units:")
        self.units_slider = QSlider(Qt.Horizontal)
        self.units_slider.setRange(10, 200)
        self.units_slider.setValue(50)
        self.units_slider.setToolTip("Number of neurons in the LSTM layer.")
        self.units_value_label = QLabel("50")
        self.units_slider.valueChanged.connect(lambda: self.units_value_label.setText(str(self.units_slider.value())))
        units_layout.addWidget(units_label)
        units_layout.addWidget(self.units_slider)
        units_layout.addWidget(self.units_value_label)
        units_group.setLayout(units_layout)
        layout.addWidget(units_group)

        # Learning Rate Group
        learning_rate_group = QGroupBox("Learning Rate")
        learning_rate_layout = QHBoxLayout()
        learning_rate_label = QLabel("Learning Rate:")
        self.learning_rate_input = QDoubleSpinBox()
        self.learning_rate_input.setRange(0.000001, 1.0)
        self.learning_rate_input.setSingleStep(0.000001)
        self.learning_rate_input.setDecimals(6)
        self.learning_rate_input.setValue(0.001)
        self.learning_rate_input.setToolTip("Step size for each iteration to converge towards the optimal value.")
        learning_rate_layout.addWidget(learning_rate_label)
        learning_rate_layout.addWidget(self.learning_rate_input)
        learning_rate_group.setLayout(learning_rate_layout)
        layout.addWidget(learning_rate_group)

        # Lookback Group
        lookback_group = QGroupBox("Lookback Window Size")
        lookback_layout = QHBoxLayout()
        lookback_label = QLabel("Lookback Window Size:")
        self.lookback_slider = QSlider(Qt.Horizontal)
        self.lookback_slider.setRange(10, 60)
        self.lookback_slider.setValue(30)
        self.lookback_value_label = QLabel("30")
        self.lookback_slider.setToolTip("Number of previous time steps to consider for each prediction.")
        self.lookback_slider.valueChanged.connect(lambda: self.lookback_value_label.setText(str(self.lookback_slider.value())))
        lookback_layout.addWidget(lookback_label)
        lookback_layout.addWidget(self.lookback_slider)
        lookback_layout.addWidget(self.lookback_value_label)
        lookback_group.setLayout(lookback_layout)
        layout.addWidget(lookback_group)

        # Save and Load Buttons
        button_layout = QHBoxLayout()
        self.save_params_button = QPushButton("Save Parameters")
        self.load_params_button = QPushButton("Load Parameters")
        self.save_params_button.setToolTip("Save the current parameters as the optimal configuration.")
        self.load_params_button.setToolTip("Load previously saved optimal parameters.")
        button_layout.addWidget(self.save_params_button)
        button_layout.addWidget(self.load_params_button)

        # Add the Save and Load Buttons layout
        layout.addLayout(button_layout)

        # Connect buttons to methods
        self.save_params_button.clicked.connect(self.save_parameters)
        self.load_params_button.clicked.connect(self.load_parameters)

        tab.setLayout(layout)
        return tab

    def save_parameters(self):
        """Saves the current LSTM parameters to a JSON file."""
        parameters = {
            "epochs": self.epochs_input.value(),
            "batch_size": int(self.batch_size_input.currentText()),  # Use currentText() and convert to int
            "units": self.units_slider.value(),
            "learning_rate": self.learning_rate_input.value(),
            "lookback": self.lookback_slider.value()
        }
        try:
            with open("lstm_parameters.json", "w") as file:
                json.dump(parameters, file)
            self.output_console.append("LSTM parameters saved successfully.")
        except Exception as e:
            self.output_console.append(f"Failed to save parameters: {e}")


    def load_parameters(self):
        """Loads LSTM parameters from a JSON file and updates input fields."""
        try:
            with open("lstm_parameters.json", "r") as file:
                parameters = json.load(file)

            # Set values for each parameter
            self.epochs_input.setValue(parameters.get("epochs", 20))
            self.batch_size_input.setCurrentText(str(parameters.get("batch_size", 32)))  # Convert batch size to string
            self.units_slider.setValue(parameters.get("units", 50))
            self.learning_rate_input.setValue(parameters.get("learning_rate", 0.001))
            self.lookback_slider.setValue(parameters.get("lookback", 30))

            self.output_console.append("LSTM parameters loaded successfully.")
        except FileNotFoundError:
            self.output_console.append("No saved parameters found.")
        except Exception as e:
            self.output_console.append(f"Failed to load parameters: {e}")


    def display_data(self):
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

        # Determine which model to use
        if model_name == "LSTM":
            self.train_lstm(scaled_data, scaler)
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

    def train_lstm(self, scaled_data, scaler):
        """Trains the LSTM model with the current parameters."""
        epochs = self.epochs_input.value()
        batch_size = int(self.batch_size_input.currentText())
        units = self.units_slider.value()

        learning_rate = self.learning_rate_input.value()
        lookback = 30  # You can add a slider or input for this if needed
        
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
        """Displays the final forecast with dates excluding weekends."""
        start_date = datetime.now() + timedelta(days=1)
        prediction_dates = get_next_business_days(start_date, num_days=len(forecast))
        
        self.output_console.append("Predicted 5-Day Forecast (excluding weekends):")
        for date, value in zip(prediction_dates, forecast):
            self.output_console.append(f"{date.strftime('%Y-%m-%d')}: {value:.2f}")
        
        if mse is not None:
            self.output_console.append(f"\nMSE: {mse:.4f}\n")
        self.progress_bar.setValue(100)

    def clear_output(self):
        """Clears the Output Console."""
        self.output_console.clear()

    def copy_output(self):
        """Copies the content of the Output Console to the clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.output_console.toPlainText())

def get_next_business_days(start_date, num_days=5):
    """Generate the next business days, skipping Saturdays and Sundays."""
    business_days = []
    current_date = start_date
    while len(business_days) < num_days:
        if current_date.weekday() < 5:  # Monday=0, Sunday=6
            business_days.append(current_date)
        current_date += timedelta(days=1)
    return business_days

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
