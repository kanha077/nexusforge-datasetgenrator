import requests
import yaml
import time
import os

class OllamaManager:
    """Checks the health of the locally running Ollama instance."""
    def __init__(self, config_path="configs/models.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.config = config.get('ollama', {})
            
        self.host = self.config.get('host', '127.0.0.1')
        self.port = self.config.get('port', 11434)
        self.model = self.config.get('model_name', 'qwen2.5:3b-instruct-q4_K_M')

    def check_health(self) -> bool:
        """Check if Ollama server is accessible."""
        url = f"http://{self.host}:{self.port}/api/tags"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"Ollama server is accessible on {self.host}:{self.port}")
                
                # Optionally check if the model is downloaded
                data = response.json()
                models = [m['name'] for m in data.get('models', [])]
                if self.model not in models and f"{self.model}:latest" not in models:
                    print(f"Warning: Model {self.model} not found in Ollama tags. Make sure you run 'ollama run {self.model}'")
                
                return True
            return False
        except requests.exceptions.RequestException as e:
            print(f"Ollama server is not accessible on {url}. Make sure it is running. Error: {e}")
            return False
