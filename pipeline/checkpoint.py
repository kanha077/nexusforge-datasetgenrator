import json
import os
from typing import Dict, Any

class CheckpointManager:
    def __init__(self, filepath: str = "cache/checkpoint.json"):
        self.filepath = filepath
        self.state = {"processed_chunks": [], "exported_samples": 0}
        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                self.state = json.load(f)

    def save(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, 'w') as f:
            json.dump(self.state, f, indent=2)

    def is_chunk_processed(self, chunk_id: str) -> bool:
        return chunk_id in self.state["processed_chunks"]

    def mark_chunk_processed(self, chunk_id: str):
        if chunk_id not in self.state["processed_chunks"]:
            self.state["processed_chunks"].append(chunk_id)
            self.save()
            
    def update_exported_samples(self, count: int):
        self.state["exported_samples"] = count
        self.save()
