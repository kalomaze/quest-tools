import requests
import json
from PyQt5.QtCore import QThread, pyqtSignal, QMutexLocker, QMutex

class ModelLoader(QThread):
    models_loaded = pyqtSignal(list)
    error_signal = pyqtSignal(str)
    
    def __init__(self, api_key, base_url):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        
    def run(self):
        try:
            print("[ModelLoader] Starting model loading process...")
            headers = {"Authorization": f"Bearer {self.api_key}"}
            url = f"{self.base_url}/v1/models"
            print(f"[ModelLoader] Requesting models from: {url}")
            print(f"[ModelLoader] Headers: {{'Authorization': 'Bearer ***'}}")
            
            response = requests.get(url, headers=headers)
            print(f"[ModelLoader] Response received. Status code: {response.status_code}")
            
            if response.status_code != 200:
                error_msg = f"API error: {response.status_code}"
                print(f"[ModelLoader] {error_msg}. Response content: {response.text[:200]}...")
                self.error_signal.emit(error_msg)
                return
                
            models = [m['id'] for m in response.json().get('data', [])]
            print(f"[ModelLoader] Successfully loaded {len(models)} models.")
            self.models_loaded.emit(models)
            print("[ModelLoader] Model loading completed successfully.")
            
        except Exception as e:
            error_msg = f"Connection error: {str(e)}"
            print(f"[ModelLoader] {error_msg}")
            self.error_signal.emit(error_msg)

class GenerationWorker(QThread):
    update_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    
    def __init__(self, api_key, prompt, base_url, model, max_tokens=None):
        super().__init__()
        self.api_key = api_key
        
        # Minimal XML-aware parsing
        try:
            end_tag = prompt.index('</corrupt>') + len('</corrupt>')
            self.xml_part = prompt[:end_tag].strip()
            self.user_part = prompt[end_tag:].lstrip('\n')
        except ValueError:
            parts = prompt.split('\n\n', 1)
            self.xml_part = parts[0] if len(parts) > 0 else ''
            self.user_part = parts[1] if len(parts) > 1 else ''

        self.base_url = base_url.rstrip('/')
        self.model = model
        self.max_tokens = max_tokens
        self._stop = False
        self._mutex = QMutex()

    def stop(self):
        with QMutexLocker(self._mutex):
            print("[GenerationWorker] Stop signal received.")
            self._stop = True

    def _should_stop(self):
        with QMutexLocker(self._mutex):
            return self._stop

    def run(self):
        try:
            print("[GenerationWorker] Starting generation process...")
            full_prompt = f"{self.xml_part}\n\n{self.user_part}"
            print(f"[GenerationWorker] Full prompt (abbreviated): {full_prompt[:200]}...")
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            sanitized_headers = {
                "Authorization": "Bearer ***",
                "Content-Type": "application/json"
            }
            url = f"{self.base_url}/v1/completions"
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": True,
                "max_tokens": self.max_tokens or 256,
                "temperature": 1.0
            }
            print(f"[GenerationWorker] Sending POST request to: {url}")
            print(f"[GenerationWorker] Headers: {sanitized_headers}")
            print(f"[GenerationWorker] Payload: model={payload['model']}, max_tokens={payload['max_tokens']}, "
                  f"temperature={payload['temperature']}, stream=True")
            
            with requests.post(
                url,
                json=payload,
                headers=headers,
                stream=True
            ) as response:
                print(f"[GenerationWorker] Response status code: {response.status_code}")
                
                if response.status_code != 200:
                    error_msg = f"API error {response.status_code}"
                    print(f"[GenerationWorker] {error_msg}. Response content: {response.text[:200]}...")
                    self.error_signal.emit(error_msg)
                    return
                
                print("[GenerationWorker] Starting response stream...")
                for line in response.iter_lines():
                    if self._should_stop():
                        print("[GenerationWorker] Stream stopped by user request.")
                        break
                    
                    if line:
                        try:
                            line_decoded = line.decode('utf-8')
                            print(f"[GenerationWorker] Received data: {line_decoded[:200]}...")
                            
                            if line_decoded.startswith('data: '):
                                chunk = line_decoded[6:].strip()
                                if chunk == '[DONE]':
                                    print("[GenerationWorker] Received [DONE] signal. Ending stream.")
                                    break
                                    
                                print(f"[GenerationWorker] Processing chunk: {chunk[:200]}...")
                                data = json.loads(chunk)
                                text = data.get('choices', [{}])[0].get('text', '')
                                if text:
                                    self.update_signal.emit(text)
                        except Exception as e:
                            print(f"[GenerationWorker] Error processing line '{line}': {str(e)}")
                            continue
                
                print("[GenerationWorker] Stream processing completed.")
                self.finished_signal.emit()
                print("[GenerationWorker] Generation process finished successfully.")
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"[GenerationWorker] {error_msg}")
            self.error_signal.emit(error_msg)