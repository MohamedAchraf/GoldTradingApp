# gui/main_gui.py

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import *
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
import yfinance as yf
from src.models import run_lstm, run_arima, run_linear_regression, run_ets
from src.preprocessing import preprocess_data

# gui/main_gui.py

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
            progress = int((epoch / total_epochs) * 100)  # Calculate progress percentage
            self.progress_value.emit(progress)
        
        # Run LSTM training with the callback for progress updates and user-defined parameters
        forecast, mse = run_lstm(
            self.scaled_data, self.scaler, callback=callback,
            epochs=self.epochs, batch_size=self.batch_size, units=self.units,
            learning_rate=self.learning_rate, lookback=self.lookback
        )
        self.result.emit(forecast, mse)
        self.progress_value.emit(100)  # Set progress to 100% once completed


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Learning Dashboard")
        self.setGeometry(100, 100, 800, 600)
        
        # Main layout and tab widget
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        self.tabs = QTabWidget()
        
        # Add tabs
        self.tabs.addTab(self.create_dashboard_tab(), "Learning Dashboard")
        self.tabs.addTab(self.create_model_tab("LSTM"), "LSTM")
        self.tabs.addTab(self.create_model_tab("ARIMA"), "ARIMA")
        self.tabs.addTab(self.create_model_tab("LR"), "Linear Regression")
        self.tabs.addTab(self.create_model_tab("ETS"), "ETS")
        
        main_layout.addWidget(self.tabs)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Connect buttons to their respective functions
        self.display_data_button.clicked.connect(self.display_data)
        self.start_prediction_button.clicked.connect(self.start_prediction)
        self.clear_output_button.clicked.connect(self.clear_output)
        self.copy_output_button.clicked.connect(self.copy_output)

        # Placeholder for fetched data
        self.data = None

    def create_dashboard_tab(self):
        """Creates the main dashboard tab with model selection and data source options."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Model Selection GroupBox
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
        
        # Data Source GroupBox
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
        
        # Console Output TextEdit
        self.output_console = QTextEdit()
        self.output_console.setReadOnly(True)
        # Style for Unix-like console appearance
        self.output_console.setStyleSheet("background-color: black; color: lightgreen; font-family: Courier; font-size: 10pt;")

        
        # Output Control Buttons (Clear and Copy)
        output_control_layout = QHBoxLayout()
        self.clear_output_button = QPushButton("Clear")
        self.copy_output_button = QPushButton("Copy")
        output_control_layout.addWidget(self.clear_output_button)
        output_control_layout.addWidget(self.copy_output_button)
        
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)  # Initialize progress bar at 0%

        # Add all widgets to layout
        top_layout = QHBoxLayout()
        top_layout.addWidget(model_group)
        top_layout.addWidget(data_group)
        
        layout.addLayout(top_layout)
        layout.addLayout(button_layout)
        layout.addWidget(QLabel("Output:"))
        layout.addWidget(self.output_console)
        layout.addLayout(output_control_layout)  # Add Clear and Copy buttons below the output console
        layout.addWidget(self.progress_bar)  # Add the progress bar
        
        tab.setLayout(layout)
        return tab

    def create_model_tab(self, model_name):
        """Creates a model-specific tab with parameter controls for a given model."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Placeholder for parameter controls (to be customized for each model)
        layout.addWidget(QLabel(f"Parameter controls for {model_name} model"))
        
        tab.setLayout(layout)
        return tab

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
            # Retrieve LSTM-specific parameters
            epochs = self.epochs_spinbox.value()
            batch_size = int(self.batch_size_combo.currentText())
            units = self.units_slider.value()
            learning_rate = float(self.learning_rate_combo.currentText())
            lookback = self.lookback_slider.value()
            
            # Start LSTM training in a separate thread with user-specified parameters
            self.worker = TrainWorker(
                scaled_data, scaler, epochs=epochs, batch_size=batch_size, 
                units=units, learning_rate=learning_rate, lookback=lookback
            )
            self.worker.progress.connect(self.output_console.append)  # Connect progress to console
            self.worker.progress_value.connect(self.progress_bar.setValue)  # Update progress bar
            self.worker.result.connect(self.display_result)  # Display final result in console
            self.progress_bar.setValue(0)  # Reset progress bar
            self.worker.start()  # Start the worker thread

        elif model_name == "ARIMA":
            # Run ARIMA model directly and display results
            forecast, mse = run_arima(combined_data['Close'])
            self.display_result(forecast, mse)

        elif model_name == "LR":
            # Run Linear Regression model directly and display results
            forecast, mse = run_linear_regression(combined_data)
            self.display_result(forecast, mse)

        elif model_name == "ETS":
            # Run ETS model directly and display results
            forecast, mse = run_ets(combined_data['Close'])
            self.display_result(forecast, mse)
        else:
            self.output_console.append("Unknown model selected.")

    def display_result(self, forecast, mse):
        """Displays the final forecast and MSE after training or prediction."""
        self.output_console.append(f"Forecast: {forecast}")
        self.output_console.append(f"MSE: {mse}\n")
        self.progress_bar.setValue(100)  # Set progress to 100% upon completion





    def display_result(self, forecast, mse=None):
        """Displays the forecast with dates in the Output Console in a user-friendly format."""
        
        # Get the last date in the data and calculate the next 5 prediction dates
        last_date = self.data.index[-1]  # Assumes self.data has a datetime index
        prediction_dates = pd.date_range(last_date, periods=len(forecast) + 1, freq='D')[1:]  # Skip the last_date itself

        # Display the forecast with dates
        self.output_console.append("Forecast:")
        for date, value in zip(prediction_dates, forecast):
            # Ensure `value` is a scalar by extracting it if it's an array
            if isinstance(value, np.ndarray):
                value = value.item()  # Convert single-element array to a scalar
            self.output_console.append(f"{date.strftime('%Y-%m-%d')}: {value:.2f}")

        # Optionally display MSE if needed
        if mse is not None:
            self.output_console.append(f"\nMSE: {mse}\n")

        # Update progress bar to 100% upon completion
        self.progress_bar.setValue(100)






    def clear_output(self):
        """Clears the Output Console."""
        self.output_console.clear()

    def copy_output(self):
        """Copies the content of the Output Console to the clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.output_console.toPlainText())

    # gui/main_gui.py


    def create_model_tab(self, model_name):
        """Creates a model-specific tab with parameter controls for a given model."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        if model_name == "LSTM":
            # Title Label
            layout.addWidget(QLabel(f"Parameter controls for {model_name} model"))
            
            # Epochs
            epochs_label = QLabel("Epochs:")
            self.epochs_spinbox = QSpinBox()
            self.epochs_spinbox.setRange(1, 100)  # Epochs from 1 to 100
            self.epochs_spinbox.setValue(10)  # Default value
            layout.addWidget(epochs_label)
            layout.addWidget(self.epochs_spinbox)
            
            # Batch Size
            batch_size_label = QLabel("Batch Size:")
            self.batch_size_combo = QComboBox()
            self.batch_size_combo.addItems(["1", "16", "32", "64"])  # Common batch sizes
            layout.addWidget(batch_size_label)
            layout.addWidget(self.batch_size_combo)
            
            # LSTM Units
            units_label = QLabel("LSTM Units:")
            self.units_slider = QSlider(Qt.Horizontal)
            self.units_slider.setRange(10, 200)  # Units from 10 to 200
            self.units_slider.setValue(50)  # Default value
            self.units_value_label = QLabel("50")
            self.units_slider.valueChanged.connect(lambda: self.units_value_label.setText(str(self.units_slider.value())))
            layout.addWidget(units_label)
            layout.addWidget(self.units_slider)
            layout.addWidget(self.units_value_label)
            
            # Learning Rate
            learning_rate_label = QLabel("Learning Rate:")
            self.learning_rate_combo = QComboBox()
            self.learning_rate_combo.addItems(["0.0001", "0.001", "0.01"])  # Common learning rates
            layout.addWidget(learning_rate_label)
            layout.addWidget(self.learning_rate_combo)
            
            # Lookback Window Size
            lookback_label = QLabel("Lookback Window Size:")
            self.lookback_slider = QSlider(Qt.Horizontal)
            self.lookback_slider.setRange(10, 60)  # Lookback from 10 to 60
            self.lookback_slider.setValue(30)  # Default value
            self.lookback_value_label = QLabel("30")
            self.lookback_slider.valueChanged.connect(lambda: self.lookback_value_label.setText(str(self.lookback_slider.value())))
            layout.addWidget(lookback_label)
            layout.addWidget(self.lookback_slider)
            layout.addWidget(self.lookback_value_label)
        
        tab.setLayout(layout)
        return tab


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
