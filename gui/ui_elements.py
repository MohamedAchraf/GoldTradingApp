from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton, QSpinBox, QSlider, QComboBox, QTextEdit, QRadioButton, QCheckBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

def create_dashboard_tab(main_window):
    # Define layout and widgets here (as in the main script)
    # and use `main_window` attributes or methods if needed for connections.
    tab = QWidget()
    layout = QVBoxLayout()
    # Model and data selection setup omitted for brevity
    return tab

def create_lstm_tab(main_window):
    # Define the LSTM-specific controls (as in the main script)
    tab = QWidget()
    layout = QVBoxLayout()
    # Setup for epochs, batch size, units, etc., omitted for brevity
    return tab
