import sys
import json
import random
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QPushButton, QLabel, QTextEdit,
                           QFileDialog, QSplitter)
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
import difflib
from itertools import zip_longest

class RestorationDiffViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Restoration Diff Viewer")
        self.setGeometry(100, 100, 1200, 800)
        self.setup_ui()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Top controls
        controls = QHBoxLayout()
        self.load_button = QPushButton("Load JSONL")
        self.load_button.clicked.connect(self.load_jsonl)
        controls.addWidget(self.load_button)
        
        self.export_button = QPushButton("Export Training Samples")
        self.export_button.clicked.connect(self.export_training_samples)
        controls.addWidget(self.export_button)
        
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous)
        controls.addWidget(self.prev_button)
        
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next)
        controls.addWidget(self.next_button)
        
        self.index_label = QLabel("0/0")
        controls.addWidget(self.index_label)
        
        controls_widget = QWidget()
        controls_widget.setLayout(controls)
        main_layout.addWidget(controls_widget)

        # Create splitter for main content
        splitter = QSplitter(Qt.Vertical)
        
        # Upper diff view
        self.diff_view = QTextEdit()
        self.diff_view.setReadOnly(True)
        splitter.addWidget(self.diff_view)

        # Lower training sample view
        self.training_view = QTextEdit()
        self.training_view.setReadOnly(True)
        splitter.addWidget(self.training_view)

        # Add splitter to main layout
        main_layout.addWidget(splitter)

        # Match percentage at bottom
        self.match_percentage = QLabel("Matching: 0%")
        main_layout.addWidget(self.match_percentage)

        # Initialize state
        self.current_index = 0
        self.restorations = []
        self.update_buttons()

    def export_training_samples(self):
        if not self.restorations:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Training Samples",
            "",
            "JSONL Files (*.jsonl);;All Files (*)"
        )
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                for entry in self.restorations:
                    training_sample = self.generate_training_sample(
                        entry.get('original', ''),
                        entry.get('restoration', '')
                    )
                    if training_sample:
                        # Split the training sample into content and judgment
                        parts = training_sample.split("SAMPLE REGION JUDGMENT:")
                        if len(parts) == 2:
                            content = parts[0].strip()
                            judgment = parts[1].strip()
                            
                            # Create JSON object
                            json_obj = {
                                "content": content,
                                "judgment": judgment
                            }
                            
                            # Write to file
                            f.write(json.dumps(json_obj) + '\n')

    def generate_training_sample(self, original, restoration):
        def find_continuous_diff_region(text1, text2):
            matcher = difflib.SequenceMatcher(None, text1, text2)
            matches = list(matcher.get_matching_blocks())
            
            # Filter out tiny matches that occur between real differences
            significant_matches = [
                (i, match) for i, match in enumerate(matches)
                if match.size > 30  # Significant matching section
            ]
            
            if len(significant_matches) < 2:
                return None
                    
            # Find the longest diff section between significant matches
            longest_diff = (0, 0, 0, 0)  # (start1, end1, start2, end2)
            max_diff_size = 0
            
            for i in range(len(significant_matches) - 1):
                curr_match = significant_matches[i][1]
                next_match = significant_matches[i + 1][1]
                
                # Calculate the size of the differing section
                diff_size1 = next_match.a - (curr_match.a + curr_match.size)
                diff_size2 = next_match.b - (curr_match.b + curr_match.size)
                total_diff = diff_size1 + diff_size2
                
                if total_diff > max_diff_size:
                    max_diff_size = total_diff
                    longest_diff = (
                        curr_match.a + curr_match.size,
                        next_match.a,
                        curr_match.b + curr_match.size,
                        next_match.b
                    )
                
            return {
                'orig': (longest_diff[0], longest_diff[1]),
                'rest': (longest_diff[2], longest_diff[3])
            }

        def find_word_boundary(text, pos, direction='forward'):
            """Find the nearest word boundary in the specified direction."""
            if direction == 'forward':
                # Look for the next space or punctuation
                i = pos
                while i < len(text) and text[i].isalnum():
                    i += 1
                return i
            else:
                # Look for the previous space or punctuation
                i = pos - 1
                while i >= 0 and text[i].isalnum():
                    i -= 1
                return i + 1

        region = find_continuous_diff_region(original, restoration)
        if region:
            # Valid difference region found - use XML tags
            orig_start = find_word_boundary(original, region['orig'][0], 'backward')
            orig_end = find_word_boundary(original, region['orig'][1], 'forward')
            rest_start = find_word_boundary(restoration, region['rest'][0], 'backward')
            rest_end = find_word_boundary(restoration, region['rest'][1], 'forward')

            def insert_region_tag(text, start, end):
                return text[:start] + '<REGIONAL_SECTION>' + text[start:end] + '</REGIONAL_SECTION>' + text[end:]

            orig_tagged = insert_region_tag(original, orig_start, orig_end)
            rest_tagged = insert_region_tag(restoration, rest_start, rest_end)
            
            if random.random() < 0.5:
                sample_a = orig_tagged
                sample_b = rest_tagged
                judgment = "A"
            else:
                sample_a = rest_tagged
                sample_b = orig_tagged
                judgment = "B"
        else:
            # No valid difference region - use texts without XML tags
            if random.random() < 0.5:
                sample_a = original
                sample_b = restoration
                judgment = "A"
            else:
                sample_a = restoration
                sample_b = original
                judgment = "B"

        return f"""SAMPLE A:
    {sample_a}

    SAMPLE B:
    {sample_b}

    SAMPLE REGION JUDGMENT: {judgment}"""

        return None

    def load_jsonl(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select JSONL file",
            "",
            "JSONL Files (*.jsonl);;All Files (*)"
        )
        
        if filename:
            self.restorations = []
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        self.restorations.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            
            self.current_index = 0
            self.update_display()
            self.update_buttons()

    def update_buttons(self):
        total = len(self.restorations)
        self.index_label.setText(f"{self.current_index + 1}/{total}" if total > 0 else "0/0")
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < total - 1)
        self.export_button.setEnabled(total > 0)

    def show_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()
            self.update_buttons()

    def show_next(self):
        if self.current_index < len(self.restorations) - 1:
            self.current_index += 1
            self.update_display()
            self.update_buttons()

    def update_display(self):
        if not self.restorations:
            return

        entry = self.restorations[self.current_index]
        original = entry.get('original', '')
        restoration = entry.get('restoration', '')

        # Calculate match percentage
        if original and restoration:
            matcher = difflib.SequenceMatcher(None, original, restoration)
            match_ratio = matcher.ratio() * 100
            self.match_percentage.setText(f"Matching: {match_ratio:.1f}%")
            
            if match_ratio > 90:
                self.match_percentage.setStyleSheet("color: green; font-weight: bold;")
            elif match_ratio > 70:
                self.match_percentage.setStyleSheet("color: orange; font-weight: bold;")
            else:
                self.match_percentage.setStyleSheet("color: red; font-weight: bold;")
        
        # Generate diff view
        original_lines = original.split('\n')
        restoration_lines = restoration.split('\n')
        
        original_html_lines = []
        restoration_html_lines = []
        
        for orig_line, rest_line in zip_longest(original_lines, restoration_lines, fillvalue=''):
            orig_words = orig_line.split()
            rest_words = rest_line.split()
            
            diff = list(difflib.Differ().compare(rest_words, orig_words))
            
            line_orig_html = []
            line_rest_html = []
            
            for token in diff:
                if token.startswith('+ '):
                    line_orig_html.append(f'<span style="background-color: rgba(73, 156, 84, 0.7); color: white;">{token[2:]}</span>')
                    line_rest_html.append('&nbsp;')
                elif token.startswith('- '):
                    line_orig_html.append('&nbsp;')
                    line_rest_html.append(f'<span style="background-color: rgba(255, 100, 100, 0.7); color: white;">{token[2:]}</span>')
                elif token.startswith('  '):
                    line_orig_html.append(token[2:])
                    line_rest_html.append(token[2:])
            
            original_html_lines.append(' '.join(line_orig_html))
            restoration_html_lines.append(' '.join(line_rest_html))

        diff_html = f"""
            <div style="font-family: Arial, sans-serif; background-color: #1e1e1e; color: #E0E0E0;">
                <div style="margin-bottom: 20px;">
                    <div style="background-color: #2d2d2d; color: white; padding: 5px 10px; border-radius: 5px 5px 0 0;">Original Text</div>
                    <div style="padding: 10px; border-radius: 0 0 5px 5px; line-height: 1.5;">
                        {('<br>'.join(original_html_lines))}
                    </div>
                </div>
                <div>
                    <div style="background-color: #2d2d2d; color: white; padding: 5px 10px; border-radius: 5px 5px 0 0;">Restoration Text</div>
                    <div style="padding: 10px; border-radius: 0 0 5px 5px; line-height: 1.5;">
                        {('<br>'.join(restoration_html_lines))}
                    </div>
                </div>
            </div>
        """
        
        self.diff_view.setHtml(diff_html)

        # Generate and display training sample
        training_sample = self.generate_training_sample(original, restoration)
        if training_sample:
            # Escape < and > to make XML tags visible as text
            escaped_sample = training_sample.replace('<', '&lt;').replace('>', '&gt;')
            
            training_html = f"""
                <div style="font-family: monospace; background-color: #1e1e1e; color: #E0E0E0;">
                    <div style="background-color: #2d2d2d; color: white; padding: 5px 10px; border-radius: 5px 5px 0 0;">Training Sample</div>
                    <div style="padding: 10px; border-radius: 0 0 5px 5px;">
                        <pre style="white-space: pre-wrap; margin: 0; font-family: 'Courier New', monospace; line-height: 1.5;">{escaped_sample}</pre>
                    </div>
                </div>
            """
            self.training_view.setHtml(training_html)
        else:
            self.training_view.setHtml(
                '<div style="font-family: monospace; background-color: #1e1e1e; color: #666; padding: 10px; border-radius: 5px;">'
                'No suitable training sample could be generated for this pair.'
                '</div>'
            )

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RestorationDiffViewer()
    window.show()
    sys.exit(app.exec_())