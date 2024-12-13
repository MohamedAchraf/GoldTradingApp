# gui/arima_tab.py

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QComboBox,
    QPushButton, QTextEdit
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd

class OptimizeARIMAThread(QThread):
    """Thread to find the optimal ARIMA order."""
    progress_signal = pyqtSignal(str)  # Signal to update the console with progress

    def __init__(self, data, arima_tab):
        super().__init__()
        self.data = data
        self.arima_tab = arima_tab  # Reference to ARIMATab to set order values

    def run(self):
        best_order = None
        best_mse = float("inf")

        # Define ranges for p, d, and q
        p_range = range(0, 5)
        d_range = range(0, 5)
        q_range = range(0, 5)

        for p in p_range:
            for d in d_range:
                for q in q_range:
                    try:
                        model = ARIMA(self.data, order=(p, d, q))
                        model_fit = model.fit()
                        forecast = model_fit.forecast(steps=5)
                        mse = mean_squared_error(self.data[-5:], forecast)

                        # Send progress to console
                        self.progress_signal.emit(f"Order (p,d,q)=({p},{d},{q}) -> MSE: {mse:.4f}")

                        if mse < best_mse:
                            best_mse = mse
                            best_order = (p, d, q)

                    except Exception as e:
                        # Send error to console without stopping
                        self.progress_signal.emit(f"Failed to fit ARIMA({p},{d},{q}): {e}")

        if best_order:
            self.progress_signal.emit(f"\nOptimal Order: (p,d,q)={best_order} with MSE: {best_mse:.4f}")
            # Set the optimal order for selection
            self.arima_tab.set_order_values(*best_order)  # Use the reference to arima_tab
        else:
            self.progress_signal.emit("\nFailed to find an optimal order.")

class ARIMATab(QWidget):
    """ARIMA tab with parameter selection and optimization functionality."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None  # Placeholder for data
        self.init_ui()

    def init_ui(self):
        """Initializes the UI components."""
        layout = QVBoxLayout()

        # Order Selection GroupBox
        order_group = QGroupBox("Order Selection (p, d, q)")
        order_layout = QHBoxLayout()
        self.p_combo = QComboBox()
        self.d_combo = QComboBox()
        self.q_combo = QComboBox()
        for i in range(5):  # Extended range for demonstration purposes
            self.p_combo.addItem(str(i))
            self.d_combo.addItem(str(i))
            self.q_combo.addItem(str(i))
        order_layout.addWidget(QLabel("p:"))
        order_layout.addWidget(self.p_combo)
        order_layout.addWidget(QLabel("d:"))
        order_layout.addWidget(self.d_combo)
        order_layout.addWidget(QLabel("q:"))
        order_layout.addWidget(self.q_combo)  # Ensure q_combo is added to the layout
        order_group.setLayout(order_layout)

        # Output Console
        self.output_console = QTextEdit()
        self.output_console.setReadOnly(True)
        self.output_console.setStyleSheet("background-color: black; color: lightgreen; font-family: Courier; font-size: 10pt;")

        # Optimize Order Button
        self.optimize_button = QPushButton("Optimize Order")
        self.optimize_button.setIcon(QIcon("assets/icons/search.svg"))  # Choose an appropriate icon
        self.optimize_button.clicked.connect(self.optimize_order)

        # Add components to main layout
        layout.addWidget(order_group)
        layout.addWidget(self.output_console)
        layout.addWidget(self.optimize_button)
        self.setLayout(layout)

    def set_data(self, data):
        """Sets the data for ARIMA analysis."""
        self.data = data
        self.output_console.append("Data has been set for ARIMA.")

    def get_order(self):
        """Retrieves selected (p, d, q) order values."""
        return int(self.p_combo.currentText()), int(self.d_combo.currentText()), int(self.q_combo.currentText())

    def set_order_values(self, p, d, q):
        """Sets the optimal values in the combo boxes for user selection."""
        self.p_combo.setCurrentText(str(p))
        self.d_combo.setCurrentText(str(d))
        self.q_combo.setCurrentText(str(q))

    def optimize_order(self):
        """Starts the ARIMA order optimization process."""
        # Clear previous output
        self.output_console.clear()

        # Check if data exists and is a valid DataFrame/Series
        if self.data is None or self.data.empty:
            self.output_console.append("No data available. Please fetch data first.")
            return

        # Extract 'Close' column for ARIMA optimization if it’s a DataFrame with that column
        series_data = self.data['Close'] if 'Close' in self.data.columns else self.data

        # Initialize and start the optimization thread
        self.optimize_thread = OptimizeARIMAThread(series_data, self)  # Pass reference to self
        self.optimize_thread.progress_signal.connect(self.output_console.append)  # Connect progress to console
        self.optimize_thread.start()
