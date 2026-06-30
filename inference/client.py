import requests
import yaml
import time
from typing import Dict, Any, Optional, List

class OllamaClient:
    def __init__(self, config_path="configs/models.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.config = config.get('ollama', {})
            
        self.host = self.config.get('host', '127.0.0.1')
        self.port = self.config.get('port', 11434)
        self.model = self.config.get('model_name', 'qwen2.5:3b-instruct-q4_K_M')
        self.base_url = f"http://{self.host}:{self.port}/api/generate"
        self.chat_url = f"http://{self.host}:{self.port}/api/chat"

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1024, retries: int = 3) -> str:
        """Generate a response using Ollama API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        if system_prompt:
            payload["system"] = system_prompt
            
        for attempt in range(retries):
            try:
                response = requests.post(self.base_url, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                return result.get('response', '').strip()
            except requests.exceptions.RequestException as e:
                print(f"Inference error (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2)
        return ""
