import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QPushButton, QCheckBox,
                           QLabel, QSpinBox, QComboBox, QLineEdit,
                           QTextEdit)
from PyQt5.QtCore import QTimer
from PyQt5 import QtGui
import pyqtgraph as pg
from datetime import datetime
from api_bullshit import ModelLoader, GenerationWorker
from text_corruptor_utils import (ResizableTextEdit, save_application_state, 
                                load_application_state, generate_corruption_pattern,
                                corrupt_text_segment, AUTOSAVE_INTERVAL, 
                                DEFAULT_API_KEY, DEFAULT_OBJECTIVE)

from itertools import zip_longest

class TextCorruptor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Text Corruptor")
        self.setGeometry(100, 100, 1400, 800)
        self.worker = None
        self.model_loader = None
        self.token_buffer = ""
        self.setup_ui()
        self.setup_autosave()
        self.load_saved_state()

    def closeEvent(self, event):
        self.stop_generation()
        if self.model_loader and self.model_loader.isRunning():
            self.model_loader.quit()
            self.model_loader.wait()
        super().closeEvent(event)

    def setup_autosave(self):
        self.autosave_timer = QTimer()
        self.autosave_timer.setInterval(AUTOSAVE_INTERVAL)
        self.autosave_timer.timeout.connect(self.save_state)
        self.autosave_timer.start()

    def save_state(self):
        state = {
            'api_settings': {
                'api_url': self.server_url.text(),
                'api_key': self.api_key.text()
            },
            'corrupted_text': self.output_text.toPlainText(),
            'input_text': self.input_text.toPlainText(),
            'generated_text': self.generated_text.toPlainText(),
            'objective_text': self.objective_text.toPlainText(),
            'corruption_settings': {
                'pattern_type': self.pattern_type.currentText(),
                'range_min': self.range_min.value(),
                'range_max': self.range_max.value(),
                'alpha_mode': self.alpha_mode.isChecked()
            },
            'timestamp': datetime.now().isoformat()
        }
        save_application_state(state)

    def load_saved_state(self):
        state = load_application_state()
        if state:
            api_settings = state.get('api_settings', {})
            self.server_url.setText(api_settings.get('api_url', 'https://api.example.link/'))
            self.api_key.setText(api_settings.get('api_key', DEFAULT_API_KEY))
            
            self.output_text.setPlainText(state.get('corrupted_text', ''))
            self.input_text.setPlainText(state.get('input_text', ''))
            self.generated_text.setPlainText(state.get('generated_text', '<original>\n'))
            self.objective_text.setPlainText(state.get('objective_text', DEFAULT_OBJECTIVE))
            
            corruption_settings = state.get('corruption_settings', {})
            if corruption_settings:
                self.pattern_type.setCurrentText(corruption_settings.get('pattern_type', 'Constant'))
                self.range_min.setValue(corruption_settings.get('range_min', 95))
                self.range_max.setValue(corruption_settings.get('range_max', 100))
                self.alpha_mode.setChecked(corruption_settings.get('alpha_mode', False))
            
            self.update_diff_view()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Left Panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # API Configuration
        api_header = QLabel("API Configuration:")
        server_widget = QWidget()
        server_layout = QHBoxLayout(server_widget)
        server_layout.addWidget(QLabel("API URL:"))
        self.server_url = QLineEdit("https://api.example.link/")
        server_layout.addWidget(self.server_url)
        
        key_widget = QWidget()
        key_layout = QHBoxLayout(key_widget)
        key_layout.addWidget(QLabel("API Key:"))
        self.api_key = QLineEdit(DEFAULT_API_KEY)
        self.api_key.setEchoMode(QLineEdit.Password)
        key_layout.addWidget(self.api_key)
        
        # Model Selection
        model_widget = QWidget()
        model_layout = QHBoxLayout(model_widget)
        model_layout.addWidget(QLabel("Model:"))
        self.model_select = QComboBox()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.load_models)
        model_layout.addWidget(self.model_select)
        model_layout.addWidget(self.refresh_btn)

        model_layout.addWidget(QLabel("Max Tokens:"))
        self.max_tokens_input = QSpinBox()
        self.max_tokens_input.setRange(0, 32000)
        self.max_tokens_input.setValue(2000)
        self.max_tokens_input.setSpecialValueText("No limit")
        model_layout.addWidget(self.max_tokens_input)

        # Corruption Settings
        corrupt_header = QLabel("Corruption Settings:")
        pattern_widget = QWidget()
        pattern_layout = QHBoxLayout(pattern_widget)
        pattern_layout.addWidget(QLabel("Pattern:"))
        self.pattern_type = QComboBox()
        self.pattern_type.addItems(['Constant', 'Gaussian', 'Linear'])
        pattern_layout.addWidget(self.pattern_type)
        
        range_widget = QWidget()
        range_layout = QHBoxLayout(range_widget)
        self.range_min = QSpinBox()
        self.range_min.setRange(0, 100)
        self.range_min.setValue(95)
        self.range_max = QSpinBox()
        self.range_max.setRange(0, 100)
        self.range_max.setValue(100)
        range_layout.addWidget(QLabel("Range:"))
        range_layout.addWidget(self.range_min)
        range_layout.addWidget(QLabel("-"))
        range_layout.addWidget(self.range_max)
        range_layout.addWidget(QLabel("%"))
        
        self.alpha_mode = QCheckBox("A-Z/0-9 Only")
        self.corrupt_btn = QPushButton("Corrupt Text")
        self.corrupt_btn.clicked.connect(self.corrupt_text)

        self.input_text = QTextEdit()
        self.input_text.textChanged.connect(self.update_diff_view)

        # Add widgets to left layout
        left_layout.addWidget(api_header)
        left_layout.addWidget(server_widget)
        left_layout.addWidget(key_widget)
        left_layout.addWidget(model_widget)
        left_layout.addWidget(QLabel("Input Text:"))
        left_layout.addWidget(self.input_text)
        left_layout.addWidget(corrupt_header)
        left_layout.addWidget(pattern_widget)
        left_layout.addWidget(range_widget)
        left_layout.addWidget(self.alpha_mode)
        left_layout.addWidget(self.corrupt_btn)

        # Middle Panel (previously right panel)
        middle_panel = QWidget()
        middle_layout = QVBoxLayout(middle_panel)
        middle_layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget(background='w')
        self.plot_widget.showGrid(x=True, y=True)
        middle_layout.addWidget(self.plot_widget, stretch=1)

        # Text container
        text_container = QWidget()
        text_container_layout = QVBoxLayout(text_container)
        text_container_layout.setContentsMargins(0, 0, 0, 0)

        # Corrupted text section
        corrupted_section = QWidget()
        corrupted_layout = QVBoxLayout(corrupted_section)
        corrupted_layout.addWidget(QLabel("Corrupted Text:"))
        self.output_text = QTextEdit()
        corrupted_layout.addWidget(self.output_text)
        text_container_layout.addWidget(corrupted_section, stretch=1)

        # Generation section
        generation_section = QWidget()
        generation_layout = QVBoxLayout(generation_section)
        generation_layout.addWidget(QLabel("Generation:"))
        
        gen_widget = QWidget()
        gen_layout = QHBoxLayout(gen_widget)
        self.generate_btn = QPushButton("Generate")
        self.stop_btn = QPushButton("Stop")
        self.generate_btn.clicked.connect(self.start_generation)
        self.stop_btn.clicked.connect(self.stop_generation)
        self.stop_btn.setEnabled(False)
        gen_layout.addWidget(self.generate_btn)
        gen_layout.addWidget(self.stop_btn)
        
        generation_layout.addWidget(gen_widget)
        
        generation_layout.addWidget(QLabel("Objective:"))
        self.objective_text = ResizableTextEdit()
        self.objective_text.setPlainText(DEFAULT_OBJECTIVE)
        self.objective_text.setMaximumHeight(100)
        generation_layout.addWidget(self.objective_text)
        
        self.generated_text = ResizableTextEdit()
        self.generated_text.setPlainText("<original>\n")
        self.generated_text.textChanged.connect(self.update_diff_view)
        generation_layout.addWidget(self.generated_text)
        text_container_layout.addWidget(generation_section, stretch=1)

        middle_layout.addWidget(text_container, stretch=3)

        # Right Panel (new diff panel)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Add diff visualization widget
        right_layout.addWidget(QLabel("Diff View:"))
        self.diff_view = QTextEdit()
        self.diff_view.setReadOnly(True)
        right_layout.addWidget(self.diff_view)

        # Add matching percentage display
        self.match_percentage = QLabel("Matching: 0%")
        right_layout.addWidget(self.match_percentage)

        # Add all panels to main layout with equal width
        layout.addWidget(middle_panel, 33)  # Middle panel now on left
        layout.addWidget(left_panel, 33)    # Left panel now in middle
        layout.addWidget(right_panel, 33)   # Right panel stays on right

    def update_diff_view(self):
        import difflib

        # Get the input and restoration texts
        input_text = self.input_text.toPlainText().strip()
        generated_text = self.generated_text.toPlainText()
        
        # Extract restoration text from between <original> tags
        restoration_text = ""
        if "<original>" in generated_text and "</original>" in generated_text:
            restoration_text = generated_text.split("<original>")[1].split("</original>")[0].strip()

        # Split texts into lines
        input_lines = input_text.split('\n')
        restoration_lines = restoration_text.split('\n')
        
        # Calculate matching percentage
        if input_text and restoration_text:
            matcher = difflib.SequenceMatcher(None, input_text, restoration_text)
            match_ratio = matcher.ratio() * 100
            self.match_percentage.setText(f"Matching: {match_ratio:.1f}%")
            
            if match_ratio > 90:
                self.match_percentage.setStyleSheet("color: green; font-weight: bold;")
            elif match_ratio > 70:
                self.match_percentage.setStyleSheet("color: orange; font-weight: bold;")
            else:
                self.match_percentage.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.match_percentage.setText("Matching: 0%")
            self.match_percentage.setStyleSheet("")

        input_html_lines = []
        restoration_html_lines = []
        
        # Process each line
        for input_line, restoration_line in zip_longest(input_lines, restoration_lines, fillvalue=''):
            # Split each line into words
            input_words = input_line.split()
            restoration_words = restoration_line.split()
            
            # Get word-level diff for this line
            diff = list(difflib.Differ().compare(restoration_words, input_words))
            
            line_input_html = []
            line_restoration_html = []
            
            for token in diff:
                if token.startswith('+ '):
                    line_input_html.append(f'<span style="background-color: rgba(73, 156, 84, 0.7); color: white;">{token[2:]}</span>')
                    line_restoration_html.append('&nbsp;')
                elif token.startswith('- '):
                    line_input_html.append('&nbsp;')
                    line_restoration_html.append(f'<span style="background-color: rgba(255, 100, 100, 0.7); color: white;">{token[2:]}</span>')
                elif token.startswith('  '):
                    line_input_html.append(token[2:])
                    line_restoration_html.append(token[2:])
                elif token.startswith('? '):
                    continue
            
            input_html_lines.append(' '.join(line_input_html))
            restoration_html_lines.append(' '.join(line_restoration_html))

        input_diff_text = '<br>'.join(input_html_lines)
        restoration_diff_text = '<br>'.join(restoration_html_lines)
        
        final_diff_text = f"""
            <div style="font-family: Arial, sans-serif;">
                <div style="margin-bottom: 20px;">
                    <div style="background-color: #2d2d2d; color: white; padding: 5px 10px; border-radius: 5px 5px 0 0;">Input Text</div>
                    <div style="background-color: #1e1e1e; padding: 10px; border-radius: 0 0 5px 5px; line-height: 1.5;">
                        {input_diff_text}
                    </div>
                </div>
                <div>
                    <div style="background-color: #2d2d2d; color: white; padding: 5px 10px; border-radius: 5px 5px 0 0;">Restoration Text</div>
                    <div style="background-color: #1e1e1e; padding: 10px; border-radius: 0 0 5px 5px; line-height: 1.5;">
                        {restoration_diff_text}
                    </div>
                </div>
            </div>
        """
        
        self.diff_view.setHtml(final_diff_text)

        cursor = self.diff_view.textCursor()
        cursor.movePosition(QtGui.QTextCursor.Start)

    def load_models(self):
        if self.model_loader and self.model_loader.isRunning():
            return

        base_url = self.server_url.text().strip()
        api_key = self.api_key.text().strip()
        
        if not base_url or not api_key:
            self.generated_text.append("Please enter API URL and key")
            return

        self.model_select.clear()
        self.model_select.addItem("Loading...")
        
        self.model_loader = ModelLoader(api_key, base_url)
        self.model_loader.models_loaded.connect(self.update_models)
        self.model_loader.error_signal.connect(self.handle_model_error)
        self.model_loader.start()

    def update_models(self, models):
        self.model_select.clear()
        self.model_select.addItems(models)

    def handle_model_error(self, error):
        self.model_select.clear()
        self.model_select.addItem("Failed to load models")
        self.generated_text.append(error)

    def start_generation(self):
        self.token_buffer = ""
        if self.worker and self.worker.isRunning():
            self.stop_generation()

        model = self.model_select.currentText()
        if not model or model == "Loading...":
            self.load_models()
            return

        corrupt_text = self.output_text.toPlainText()
        objective_text = self.objective_text.toPlainText()
        first_line = self.generated_text.toPlainText().split('\n')[0]
        self.generated_text.setPlainText(f"{first_line}\n")
        
        prompt = f"{corrupt_text}\n\n{objective_text}\n\n{first_line}\n"
        max_tokens = self.max_tokens_input.value() or None
        
        self.worker = GenerationWorker(
            self.api_key.text().strip(),
            prompt,
            self.server_url.text().strip(),
            model,
            max_tokens
        )
        self.worker.update_signal.connect(self.append_generated_text)
        self.worker.finished_signal.connect(self.generation_finished)
        self.worker.error_signal.connect(self.handle_generation_error)
        
        self.generate_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.worker.start()

    def append_generated_text(self, text):
        self.token_buffer += text
        
        if any(tag in self.token_buffer for tag in ["<obj", "<orig", "</orig", "</object"]):
            splits = []
            if "<obj" in self.token_buffer:
                splits.append(self.token_buffer.split("<obj")[0])
            if "<orig" in self.token_buffer:
                splits.append(self.token_buffer.split("<orig")[0])
            if "</orig" in self.token_buffer:
                splits.append(self.token_buffer.split("</orig")[0])
            if "</object" in self.token_buffer:
                splits.append(self.token_buffer.split("</object")[0])
            
            if splits:
                clean_text = min(splits, key=len)
                first_line = self.generated_text.toPlainText().split('\n')[0]
                self.generated_text.setPlainText(f"{first_line}\n{clean_text.strip()}\n</original>")
                self.stop_generation()
                self.token_buffer = ""
                return

        current_text = self.generated_text.toPlainText() + text
        if len(current_text) > 100:
            lines = current_text.split('\n')
            first_line = lines[0]
            for line in lines[1:]:
                if len(line.strip()) > 50:
                    if current_text.count(line) > 1:
                        clean_text = current_text.split(line)[0] + line
                        self.generated_text.setPlainText(f"{first_line}\n{clean_text.strip()}\n</original>")
                        self.stop_generation()
                        self.token_buffer = ""
                        return

        self.generated_text.moveCursor(QtGui.QTextCursor.End)
        self.generated_text.insertPlainText(text)
        self.generated_text.ensureCursorVisible()
        self.update_diff_view()

    def generation_finished(self):
        self.generate_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def handle_generation_error(self, error):
        self.generated_text.append(f"\nError: {error}")
        self.generation_finished()

    def stop_generation(self):
        if self.worker:
            self.worker.stop()
            if not self.worker.wait(1000):
                self.worker.terminate()
                self.worker.wait()
            self.worker.deleteLater()
            self.worker = None
        self.generation_finished()

    def corrupt_text(self):
        output_cursor = self.output_text.textCursor()
        if output_cursor.hasSelection():
            full_text = self.output_text.toPlainText()
            start = output_cursor.selectionStart()
            end = output_cursor.selectionEnd()
            selected_text = full_text[start:end]
            
            pattern = generate_corruption_pattern(
                self.pattern_type.currentText(),
                len(selected_text),
                self.range_min.value(),
                self.range_max.value()
            )
            
            self.plot_widget.clear()
            self.plot_widget.plot(pattern, pen=pg.mkPen('b', width=2))
            
            corrupted = corrupt_text_segment(selected_text, pattern, self.alpha_mode.isChecked())
            new_text = full_text[:start] + corrupted + full_text[end:]
            self.output_text.setPlainText(new_text)
            return

        input_cursor = self.input_text.textCursor()
        if input_cursor.hasSelection():
            input_text = self.input_text.toPlainText()
            start = input_cursor.selectionStart()
            end = input_cursor.selectionEnd()
            selected_text = input_text[start:end]
            
            pattern = generate_corruption_pattern(
                self.pattern_type.currentText(),
                len(selected_text),
                self.range_min.value(),
                self.range_max.value()
            )
            
            self.plot_widget.clear()
            self.plot_widget.plot(pattern, pen=pg.mkPen('b', width=2))
            
            corrupted = corrupt_text_segment(selected_text, pattern, self.alpha_mode.isChecked())
            new_output = input_text[:start] + corrupted + input_text[end:]
            self.output_text.setPlainText(new_output)
            return

        input_text = self.input_text.toPlainText()
        if not input_text:
            return
        
        pattern = generate_corruption_pattern(
            self.pattern_type.currentText(),
            len(input_text),
            self.range_min.value(),
            self.range_max.value()
        )
        
        self.plot_widget.clear()
        self.plot_widget.plot(pattern, pen=pg.mkPen('b', width=2))
        
        corrupted = corrupt_text_segment(input_text, pattern, self.alpha_mode.isChecked())
        self.output_text.setPlainText(corrupted)
        self.generated_text.setPlainText("<original>\n")
        self.update_diff_view()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TextCorruptor()
    window.show()
    sys.exit(app.exec_())