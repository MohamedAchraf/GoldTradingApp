# gui/lstm_tab.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QSpinBox, QComboBox, QLabel, QSlider, QDoubleSpinBox, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt
from ui_elements import create_console_output
import json

class LSTMTab(QWidget):
    """Defines the LSTM tab."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initializes the LSTM tab UI."""
        layout = QVBoxLayout()

        # Epochs
        epochs_group = QGroupBox("Epoch Settings")
        epochs_layout = QHBoxLayout()
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 100)
        self.epochs_input.setValue(10)
        epochs_layout.addWidget(QLabel("Epochs:"))
        epochs_layout.addWidget(self.epochs_input)
        epochs_group.setLayout(epochs_layout)
        layout.addWidget(epochs_group)

        # Batch Size
        batch_size_group = QGroupBox("Batch Size")
        batch_size_layout = QHBoxLayout()
        self.batch_size_input = QComboBox()
        self.batch_size_input.addItems(["1", "16", "32", "64", "128"])
        batch_size_layout.addWidget(QLabel("Batch Size:"))
        batch_size_layout.addWidget(self.batch_size_input)
        batch_size_group.setLayout(batch_size_layout)
        layout.addWidget(batch_size_group)

        # Learning Rate
        learning_rate_group = QGroupBox("Learning Rate")
        learning_rate_layout = QHBoxLayout()
        self.learning_rate_input = QDoubleSpinBox()
        self.learning_rate_input.setRange(0.000001, 1.0)
        self.learning_rate_input.setSingleStep(0.000001)
        self.learning_rate_input.setDecimals(6)
        learning_rate_layout.addWidget(QLabel("Learning Rate:"))
        learning_rate_layout.addWidget(self.learning_rate_input)
        learning_rate_group.setLayout(learning_rate_layout)
        layout.addWidget(learning_rate_group)

        # Save and Load Buttons
        button_layout = QHBoxLayout()
        self.save_params_button = QPushButton("Save Parameters")
        self.load_params_button = QPushButton("Load Parameters")
        button_layout.addWidget(self.save_params_button)
        button_layout.addWidget(self.load_params_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def save_parameters(self):
        parameters = {
            "epochs": self.epochs_input.value(),
            "batch_size": int(self.batch_size_input.currentText()),
            "learning_rate": self.learning_rate_input.value()
        }
        with open("lstm_parameters.json", "w") as file:
            json.dump(parameters, file)

    def load_parameters(self):
        try:
            with open("lstm_parameters.json", "r") as file:
                parameters = json.load(file)
            self.epochs_input.setValue(parameters["epochs"])
            self.batch_size_input.setCurrentText(str(parameters["batch_size"]))
            self.learning_rate_input.setValue(parameters["learning_rate"])
        except FileNotFoundError:
            pass
