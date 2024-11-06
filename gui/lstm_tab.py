# gui/lstm_tab.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QLabel, QSpinBox, QSlider, QDoubleSpinBox, QComboBox, QHBoxLayout
from PyQt5.QtCore import Qt

class LSTMTab(QWidget):
    """LSTM Tab with user controls for LSTM model parameters."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize the UI layout for LSTM parameters."""
        layout = QVBoxLayout()

        # Epochs group
        epochs_group = QGroupBox("Epoch Settings")
        epochs_layout = QHBoxLayout()
        epochs_label = QLabel("Epochs:")
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 100)
        self.epochs_input.setValue(20)
        epochs_layout.addWidget(epochs_label)
        epochs_layout.addWidget(self.epochs_input)
        epochs_group.setLayout(epochs_layout)
        layout.addWidget(epochs_group)

        # Batch Size group
        batch_size_group = QGroupBox("Batch Size Settings")
        batch_size_layout = QHBoxLayout()
        batch_size_label = QLabel("Batch Size:")
        self.batch_size_input = QComboBox()
        self.batch_size_input.addItems(["1", "16", "32", "64", "128"])
        batch_size_layout.addWidget(batch_size_label)
        batch_size_layout.addWidget(self.batch_size_input)
        batch_size_group.setLayout(batch_size_layout)
        layout.addWidget(batch_size_group)

        # LSTM Units group
        units_group = QGroupBox("LSTM Units")
        units_layout = QHBoxLayout()
        units_label = QLabel("LSTM Units:")
        self.units_slider = QSlider(Qt.Horizontal)
        self.units_slider.setRange(10, 200)
        self.units_slider.setValue(50)
        self.units_value_label = QLabel("50")
        self.units_slider.valueChanged.connect(lambda: self.units_value_label.setText(str(self.units_slider.value())))
        units_layout.addWidget(units_label)
        units_layout.addWidget(self.units_slider)
        units_layout.addWidget(self.units_value_label)
        units_group.setLayout(units_layout)
        layout.addWidget(units_group)

        # Learning Rate group
        learning_rate_group = QGroupBox("Learning Rate")
        learning_rate_layout = QHBoxLayout()
        learning_rate_label = QLabel("Learning Rate:")
        self.learning_rate_input = QDoubleSpinBox()
        self.learning_rate_input.setRange(0.000001, 1.0)
        self.learning_rate_input.setSingleStep(0.000001)
        self.learning_rate_input.setDecimals(6)
        self.learning_rate_input.setValue(0.001)
        learning_rate_layout.addWidget(learning_rate_label)
        learning_rate_layout.addWidget(self.learning_rate_input)
        learning_rate_group.setLayout(learning_rate_layout)
        layout.addWidget(learning_rate_group)

        # Lookback Window Size group
        lookback_group = QGroupBox("Lookback Window Size")
        lookback_layout = QHBoxLayout()
        lookback_label = QLabel("Lookback Window Size:")
        self.lookback_slider = QSlider(Qt.Horizontal)
        self.lookback_slider.setRange(10, 60)
        self.lookback_slider.setValue(30)
        self.lookback_value_label = QLabel("30")
        self.lookback_slider.valueChanged.connect(lambda: self.lookback_value_label.setText(str(self.lookback_slider.value())))
        lookback_layout.addWidget(lookback_label)
        lookback_layout.addWidget(self.lookback_slider)
        lookback_layout.addWidget(self.lookback_value_label)
        lookback_group.setLayout(lookback_layout)
        layout.addWidget(lookback_group)

        self.setLayout(layout)
