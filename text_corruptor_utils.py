import random
import numpy as np
import string
import json
import os
from datetime import datetime
from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTextEdit


# Constants
utf8_chars = [chr(i) for i in range(0x0100, 0x0800)]
alphanumeric_chars = list(string.ascii_letters + string.digits)
AUTOSAVE_INTERVAL = 10000
SAVE_FILE_PATH = "text_corruptor_state.json"
DEFAULT_API_KEY = "password"
DEFAULT_OBJECTIVE = "<objective>\ngently repair the <original> content\n</objective>"

class ResizableTextEdit(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.dragging = False
        self.start_height = self.height()
        self.start_y = self.y()
        self.setCursor(Qt.SizeVerCursor)
        self.accumulated_text = ""

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and event.pos().y() < 10:
            self.dragging = True
            self.start_height = self.height()
            self.start_y = self.y()
            self.start_pos = event.globalPos()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragging:
            delta = self.start_pos.y() - event.globalPos().y()
            new_height = max(50, self.start_height + delta)
            bottom_pos = self.start_y + self.start_height
            new_y = bottom_pos - new_height
            self.setFixedHeight(new_height)
            self.setY(new_y)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.dragging = False
        self.setFixedHeight(self.height())
        super().mouseReleaseEvent(event)

def save_application_state(state_dict):
    try:
        with open(SAVE_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(state_dict, f, ensure_ascii=False, indent=2)
        print(f"State saved at {datetime.now()}")
        return True
    except Exception as e:
        print(f"Error saving state: {e}")
        return False

def load_application_state():
    try:
        if os.path.exists(SAVE_FILE_PATH):
            with open(SAVE_FILE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading saved state: {e}")
    return None

def generate_corruption_pattern(pattern_type, length, range_min, range_max):
    x = np.linspace(0, 1, length)
    max_val = random.uniform(range_min, range_max)
    
    match pattern_type:
        case 'Constant':
            return np.full(length, max_val)
        case 'Gaussian':
            mean = random.uniform(0.3, 0.7)
            std = random.uniform(0.1, 0.3)
            return max_val * np.exp(-((x - mean)**2)/(2*std**2))
        case _:
            return max_val * (1 - x)

def corrupt_text_segment(text_segment, pattern, alpha_mode=False):
    chars = alphanumeric_chars if alpha_mode else utf8_chars
    return ''.join([
        c if c == '\n' or random.random() * 100 >= pattern[i] else random.choice(chars)
        for i, c in enumerate(text_segment)
    ])