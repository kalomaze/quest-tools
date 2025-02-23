import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QPushButton, QLabel, QTextEdit,
                           QLineEdit, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import requests
import json
import math

DEFAULT_API_KEY = "your-default-key-here"

class ModelLoader(QThread):
    models_loaded = pyqtSignal(list)
    error_signal = pyqtSignal(str)

    def __init__(self, api_key, base_url):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url

    def run(self):
        try:
            response = requests.get(
                f"{self.base_url}/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()
            models = [model["id"] for model in response.json()["data"]]
            self.models_loaded.emit(models)
        except Exception as e:
            self.error_signal.emit(str(e))

class ABComparator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("A/B Sample Comparator")
        self.setGeometry(100, 100, 1400, 800)
        self.model_loader = None
        self.setup_ui()
        self.load_state()  # Load the state when starting

    def save_state(self):
        state = {
            'server_url': self.server_url.text(),
            'sample_a': self.sample_a_text.toPlainText(),
            'sample_b': self.sample_b_text.toPlainText()
        }
        
        try:
            with open('state_rm_frontend.json', 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"Error saving state: {e}")

    def load_state(self):
        try:
            if os.path.exists('state_rm_frontend.json'):
                with open('state_rm_frontend.json', 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    self.server_url.setText(state.get('server_url', ''))
                    self.sample_a_text.setPlainText(state.get('sample_a', ''))
                    self.sample_b_text.setPlainText(state.get('sample_b', ''))
        except Exception as e:
            print(f"Error loading state: {e}")

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # API Configuration
        api_config = QWidget()
        api_layout = QVBoxLayout(api_config)
        
        # Server URL
        server_widget = QWidget()
        server_layout = QHBoxLayout(server_widget)
        server_layout.addWidget(QLabel("API URL:"))
        self.server_url = QLineEdit("http://localhost:8000")
        self.server_url.textChanged.connect(self.save_state)
        server_layout.addWidget(self.server_url)
        api_layout.addWidget(server_widget)
        
        # API Key
        key_widget = QWidget()
        key_layout = QHBoxLayout(key_widget)
        key_layout.addWidget(QLabel("API Key:"))
        self.api_key = QLineEdit(DEFAULT_API_KEY)
        self.api_key.setEchoMode(QLineEdit.Password)
        key_layout.addWidget(self.api_key)
        api_layout.addWidget(key_widget)
        
        # Model Selection
        model_widget = QWidget()
        model_layout = QHBoxLayout(model_widget)
        model_layout.addWidget(QLabel("Model:"))
        self.model_select = QComboBox()
        self.refresh_btn = QPushButton("Refresh Models")
        self.refresh_btn.clicked.connect(self.load_models)
        model_layout.addWidget(self.model_select)
        model_layout.addWidget(self.refresh_btn)
        api_layout.addWidget(model_widget)
        
        layout.addWidget(api_config)

        # Sample texts
        samples_widget = QWidget()
        samples_layout = QHBoxLayout(samples_widget)

        # Sample A
        sample_a_widget = QWidget()
        sample_a_layout = QVBoxLayout(sample_a_widget)
        sample_a_layout.addWidget(QLabel("Sample A:"))
        self.sample_a_text = QTextEdit()
        self.sample_a_text.setMinimumHeight(300)
        self.sample_a_text.setMinimumWidth(600)
        self.sample_a_text.textChanged.connect(self.save_state)
        sample_a_layout.addWidget(self.sample_a_text)
        samples_layout.addWidget(sample_a_widget)

        # Sample B
        sample_b_widget = QWidget()
        sample_b_layout = QVBoxLayout(sample_b_widget)
        sample_b_layout.addWidget(QLabel("Sample B:"))
        self.sample_b_text = QTextEdit()
        self.sample_b_text.setMinimumHeight(300)
        self.sample_b_text.setMinimumWidth(600)
        self.sample_b_text.textChanged.connect(self.save_state)
        sample_b_layout.addWidget(self.sample_b_text)
        samples_layout.addWidget(sample_b_widget)

        layout.addWidget(samples_widget)

        # Buttons container
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        
        # Swap button
        self.swap_btn = QPushButton("Swap Samples")
        self.swap_btn.clicked.connect(self.swap_samples)
        buttons_layout.addWidget(self.swap_btn)
        
        # Add Regional Tags button
        self.add_tags_btn = QPushButton("Add Regional Tags")
        self.add_tags_btn.clicked.connect(self.add_regional_tags)
        buttons_layout.addWidget(self.add_tags_btn)
        
        # Remove Regional Tags button
        self.remove_tags_btn = QPushButton("Remove Regional Tags")
        self.remove_tags_btn.clicked.connect(self.remove_regional_tags)
        buttons_layout.addWidget(self.remove_tags_btn)
        
        # Compare button
        self.compare_btn = QPushButton("Compare Samples")
        self.compare_btn.clicked.connect(self.compare_samples)
        buttons_layout.addWidget(self.compare_btn)
        
        layout.addWidget(buttons_widget)

        # Visualization boxes
        self.viz_widget = QWidget()
        viz_layout = QHBoxLayout(self.viz_widget)
        
        # Sample A visualization
        self.viz_a = QTextEdit()
        self.viz_a.setReadOnly(True)
        viz_layout.addWidget(self.viz_a)
        
        # Sample B visualization
        self.viz_b = QTextEdit()
        self.viz_b.setReadOnly(True)
        viz_layout.addWidget(self.viz_b)
        
        layout.addWidget(self.viz_widget)

        # Results display at the bottom
        self.results_label = QLabel("Results will appear here")
        self.results_label.setAlignment(Qt.AlignCenter)
        self.results_label.setWordWrap(True)
        layout.addWidget(self.results_label)

    def add_regional_tags(self):
        def insert_region_tag(text):
            return '<REGIONAL_SECTION>' + text + '</REGIONAL_SECTION>'
        
        # Add tags to both samples
        text_a = self.sample_a_text.toPlainText()
        text_b = self.sample_b_text.toPlainText()
        
        self.sample_a_text.setPlainText(insert_region_tag(text_a))
        self.sample_b_text.setPlainText(insert_region_tag(text_b))
        self.save_state()

    def remove_regional_tags(self):
        def remove_tags(text):
            text = text.replace('<REGIONAL_SECTION>', '')
            text = text.replace('</REGIONAL_SECTION>', '')
            return text
        
        # Remove tags from both samples
        text_a = self.sample_a_text.toPlainText()
        text_b = self.sample_b_text.toPlainText()
        
        self.sample_a_text.setPlainText(remove_tags(text_a))
        self.sample_b_text.setPlainText(remove_tags(text_b))
        self.save_state()

    def swap_samples(self):
        text_a = self.sample_a_text.toPlainText()
        text_b = self.sample_b_text.toPlainText()
        self.sample_a_text.setPlainText(text_b)
        self.sample_b_text.setPlainText(text_a)
        self.save_state()

    def load_models(self):
        if self.model_loader and self.model_loader.isRunning():
            return

        base_url = self.server_url.text().strip()
        api_key = self.api_key.text().strip()
        
        if not base_url or not api_key:
            self.results_label.setText("Please enter API URL and key")
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
        self.results_label.setText(f"Error: {error}")

    def compare_samples(self):
        sample_a = self.sample_a_text.toPlainText()
        sample_b = self.sample_b_text.toPlainText()

        if not sample_a or not sample_b:
            self.results_label.setText("Please enter both samples")
            return

        model = self.model_select.currentText()
        if not model or model == "Loading...":
            self.load_models()
            return

        prompt = f"""SAMPLE A:
{sample_a}

SAMPLE B:
{sample_b}

ANSWER:"""

        try:
            response = self.query_vllm(prompt)
            logprobs = self.extract_logprobs(response)
            self.update_visualization(logprobs)
        except Exception as e:
            self.results_label.setText(f"Error: {str(e)}")

    def query_vllm(self, prompt):
        url = f"{self.server_url.text().strip()}/v1/completions"
        
        data = {
            "model": self.model_select.currentText(),
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 1.0,
            "logprobs": 20
        }

        headers = {
            "Authorization": f"Bearer {self.api_key.text().strip()}"
        }

        response = requests.post(url, json=data, headers=headers)
        return response.json()

    def extract_logprobs(self, response):
        try:
            top_logprobs = response['choices'][0]['logprobs']['top_logprobs'][0]
            
            # Extract A and B logprobs
            prob_a = math.exp(top_logprobs.get(" A", float('-inf')))
            prob_b = math.exp(top_logprobs.get(" B", float('-inf')))
            
            # Normalize
            total = prob_a + prob_b
            prob_a_norm = prob_a / total * 100
            prob_b_norm = prob_b / total * 100
            
            return {
                'A': prob_a_norm,
                'B': prob_b_norm,
                'raw_logprobs': top_logprobs
            }
        except Exception as e:
            raise Exception(f"Failed to extract logprobs: {str(e)}\nResponse: {json.dumps(response, indent=2)}")

    def get_color_style(self, probability):
        """Convert probability to color style"""
        if probability > 50:
            # Green gradient
            intensity = (probability - 50) * 2
            return f"background-color: rgba(0, 255, 0, {intensity/100})"
        else:
            # Red gradient
            intensity = (50 - probability) * 2
            return f"background-color: rgba(255, 0, 0, {intensity/100})"

    def update_visualization(self, logprobs):
        prob_a = logprobs['A']
        prob_b = logprobs['B']

        # Update visualizations with colored backgrounds
        style_a = self.get_color_style(prob_a)
        style_b = self.get_color_style(prob_b)

        self.viz_a.setStyleSheet(style_a)
        self.viz_b.setStyleSheet(style_b)
        
        self.viz_a.setPlainText(f"Sample A\n{prob_a:.1f}%")
        self.viz_b.setPlainText(f"Sample B\n{prob_b:.1f}%")

        # Format the raw logprobs more cleanly
        raw_logprobs = logprobs['raw_logprobs']
        formatted_logprobs = "\n".join([
            f"{token}: {prob:.3f}" 
            for token, prob in sorted(raw_logprobs.items(), key=lambda x: x[1], reverse=True)
        ])

        # Update results label with cleaner formatting
        self.results_label.setText(
            f"Log probabilities:\n{formatted_logprobs}"
        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ABComparator()
    window.show()
    sys.exit(app.exec_())