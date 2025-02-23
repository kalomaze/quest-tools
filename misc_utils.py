from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QLabel, QSpinBox, QComboBox, QLineEdit, QTextEdit, QCheckBox)
import pyqtgraph as pg
from text_corruptor_utils import ResizableTextEdit, DEFAULT_API_KEY, DEFAULT_OBJECTIVE

def create_api_config_panel():
    api_header = QLabel("API Configuration:")
    server_widget = QWidget()
    server_layout = QHBoxLayout(server_widget)
    server_layout.addWidget(QLabel("API URL:"))
    server_url = QLineEdit("https://api.mangymango.men/")
    server_layout.addWidget(server_url)
    
    key_widget = QWidget()
    key_layout = QHBoxLayout(key_widget)
    key_layout.addWidget(QLabel("API Key:"))
    api_key = QLineEdit(DEFAULT_API_KEY)
    api_key.setEchoMode(QLineEdit.Password)
    key_layout.addWidget(api_key)
    
    return api_header, server_widget, key_widget, server_url, api_key

def create_model_selection_panel():
    model_widget = QWidget()
    model_layout = QHBoxLayout(model_widget)
    model_layout.addWidget(QLabel("Model:"))
    model_select = QComboBox()
    refresh_btn = QPushButton("Refresh")
    model_layout.addWidget(model_select)
    model_layout.addWidget(refresh_btn)

    model_layout.addWidget(QLabel("Max Tokens:"))
    max_tokens_input = QSpinBox()
    max_tokens_input.setRange(0, 32000)
    max_tokens_input.setValue(2000)
    max_tokens_input.setSpecialValueText("No limit")
    model_layout.addWidget(max_tokens_input)
    
    return model_widget, model_select, refresh_btn, max_tokens_input

def create_corruption_panel():
    corrupt_header = QLabel("Corruption Settings:")
    pattern_widget = QWidget()
    pattern_layout = QHBoxLayout(pattern_widget)
    pattern_layout.addWidget(QLabel("Pattern:"))
    pattern_type = QComboBox()
    pattern_type.addItems(['Constant', 'Gaussian', 'Linear'])
    pattern_layout.addWidget(pattern_type)
    
    range_widget = QWidget()
    range_layout = QHBoxLayout(range_widget)
    range_min = QSpinBox()
    range_min.setRange(0, 100)
    range_min.setValue(95)
    range_max = QSpinBox()
    range_max.setRange(0, 100)
    range_max.setValue(100)
    range_layout.addWidget(QLabel("Range:"))
    range_layout.addWidget(range_min)
    range_layout.addWidget(QLabel("-"))
    range_layout.addWidget(range_max)
    range_layout.addWidget(QLabel("%"))
    
    alpha_mode = QCheckBox("A-Z/0-9 Only")
    corrupt_btn = QPushButton("Corrupt Text")
    
    return (corrupt_header, pattern_widget, range_widget, pattern_type, 
            range_min, range_max, alpha_mode, corrupt_btn)

def create_text_panels():
    input_text = QTextEdit()
    output_text = QTextEdit()
    generated_text = ResizableTextEdit()
    generated_text.setPlainText("<original>\n")
    
    objective_text = ResizableTextEdit()
    objective_text.setPlainText(DEFAULT_OBJECTIVE)
    objective_text.setMaximumHeight(100)
    
    plot_widget = pg.PlotWidget(background='w')
    plot_widget.showGrid(x=True, y=True)
    
    diff_view = QTextEdit()
    diff_view.setReadOnly(True)
    
    return input_text, output_text, generated_text, objective_text, plot_widget, diff_view

def create_generation_controls():
    generate_btn = QPushButton("Generate")
    stop_btn = QPushButton("Stop")
    stop_btn.setEnabled(False)
    
    gen_widget = QWidget()
    gen_layout = QHBoxLayout(gen_widget)
    gen_layout.addWidget(generate_btn)
    gen_layout.addWidget(stop_btn)
    
    return gen_widget, generate_btn, stop_btn