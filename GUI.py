'''
Graphical User Interface for the Monkey Identification Project @cnl-DPZ
acalapai@dpz.eu - 2024
'''
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
import json
import os

# ------------------ Configuration Loading ------------------

def load_interface_config():
    config_path = os.path.join(os.path.dirname(__file__), 'branch_config.json')

    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# ------------------ Interface Building ------------------

def build_interface(config):
    current_branch = "main"  # Replace with actual branch detection

    branch_settings = config.get(current_branch)

    if branch_settings:
        layout = QVBoxLayout()  # Our main layout

        # Add buttons
        for button_name in branch_settings["buttons"]:
            button = QPushButton(button_name)
            button.clicked.connect(lambda _, btn_name=button_name: handle_button_click(btn_name))
            layout.addWidget(button)

        window.setLayout(layout)

# ------------------ Event Handling ------------------

def handle_button_click(button_name):
    print(f"Button clicked: {button_name}")

# ------------------ GUI Setup ------------------

if __name__ == "__main__":
    app = QApplication([])
    window = QWidget()
    window.setWindowTitle("M-ID")

    interface_config = load_interface_config()
    build_interface(interface_config)

    window.show()
    app.exec()
