# gui/ui_elements.py
from PyQt5.QtWidgets import QTextEdit

def create_console_output():
    """Creates a styled QTextEdit widget to display console-like output."""
    console_output = QTextEdit()
    console_output.setReadOnly(True)
    console_output.setStyleSheet("background-color: black; color: lightgreen; font-family: Courier; font-size: 10pt;")
    return console_output
