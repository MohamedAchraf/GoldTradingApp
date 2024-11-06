# gui/arima_tab.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton, QTextEdit, QGroupBox, QHBoxLayout
from ui_elements import create_console_output

class ARIMATab(QWidget):
    """Defines the ARIMA tab."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initializes the ARIMA tab UI."""
        layout = QVBoxLayout()

        # ARIMA Order Selection
        order_group = QGroupBox("Order Selection (p, d, q)")
        order_layout = QHBoxLayout()
        self.p_order = QComboBox()
        self.d_order = QComboBox()
        self.q_order = QComboBox()
        for i in range(6):
            self.p_order.addItem(str(i))
            self.d_order.addItem(str(i))
            self.q_order.addItem(str(i))
        order_layout.addWidget(QLabel("p:"))
        order_layout.addWidget(self.p_order)
        order_layout.addWidget(QLabel("d:"))
        order_layout.addWidget(self.d_order)
        order_layout.addWidget(QLabel("q:"))
        order_layout.addWidget(self.q_order)
        order_group.setLayout(order_layout)
        layout.addWidget(order_group)

        # Optimization Console
        self.output_console = create_console_output()
        self.optimize_button = QPushButton("Optimize Order")
        
        layout.addWidget(self.output_console)
        layout.addWidget(self.optimize_button)
        self.setLayout(layout)
