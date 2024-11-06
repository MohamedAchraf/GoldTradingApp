# gui/main_window.py

from PyQt5.QtWidgets import QMainWindow, QTabWidget, QApplication
from PyQt5.QtGui import QIcon
import sys
import os
from learning_dashboard_tab import LearningDashboardTab
from lstm_tab import LSTMTab
from arima_tab import ARIMATab

class MainWindow(QMainWindow):
    """Main application window."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Learning Dashboard")
        self.setGeometry(100, 100, 800, 600)
        self.init_ui()

    def init_ui(self):
        """Initializes the main window UI."""
        self.tabs = QTabWidget()

        # Initialize each tab
        self.learning_dashboard_tab = LearningDashboardTab(self)
        self.lstm_tab = LSTMTab(self)
        self.arima_tab = ARIMATab(self)

        # Add tabs to the main tab widget
        self.tabs.addTab(self.learning_dashboard_tab, "Learning Dashboard")
        self.tabs.setTabIcon(0, QIcon("assets/icons/cpu.svg"))
        self.tabs.addTab(self.lstm_tab, "LSTM")
        self.tabs.setTabIcon(1, QIcon("assets/icons/feather.svg"))
        self.tabs.addTab(self.arima_tab, "ARIMA")
        self.tabs.setTabIcon(2, QIcon("assets/icons/fingerprint.svg"))

        # Link lstm_tab to learning_dashboard_tab to access LSTM parameters
        self.learning_dashboard_tab.set_lstm_tab(self.lstm_tab)

        # Set the main widget
        self.setCentralWidget(self.tabs)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
