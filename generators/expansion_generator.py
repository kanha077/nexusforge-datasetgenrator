import uuid
import time
from typing import Dict, Any, List
from inference.client import OllamaClient
from generators.retriever import Retriever

class ExpansionGenerator:
    def __init__(self, client: OllamaClient, prompt_template: str, retriever: Retriever):
        self.client = client
        self.prompt_template = prompt_template
        self.retriever = retriever

    def generate(self, existing_samples: List[Dict[str, Any]], num_samples: int = 3, persona: str = "") -> List[Dict[str, Any]]:
        examples_text = ""
        for i, sample in enumerate(existing_samples):
            examples_text += f"Instruction: {sample['instruction']}\nOutput: {sample['output']}\n\n"
            
        prompt = self.prompt_template.format(num_samples=num_samples, examples=examples_text.strip(), persona=persona)
        response = self.client.generate(prompt=prompt, max_tokens=1024)
        
        results = []
        blocks = response.split("Instruction:")
        for block in blocks:
            if "Output:" in block:
                parts = block.split("Output:")
                instruction = parts[0].strip()
                output = parts[1].strip()
                
                combined_text = f"{instruction} {output}".lower()
                keywords = ["gym", "workout", "training", "recovery", "sleep", "deadline", "exam", "study", "gpa", "professor", "group project", "sports"]
                if not any(k in combined_text for k in keywords):
                    continue
                    
                import numpy as np
                self.retriever._lazy_load()
                source_chunk_id = None
                if self.retriever.model and len(self.retriever.embeddings) > 0:
                    sample_emb = self.retriever.model.encode([f"{instruction}\n{output}"], convert_to_numpy=True)[0]
                    similarities = np.dot(self.retriever.embeddings, sample_emb) / (
                        np.linalg.norm(self.retriever.embeddings, axis=1) * np.linalg.norm(sample_emb)
                    )
                    max_sim = np.max(similarities)
                    if max_sim < 0.2:
                        continue
                    best_idx = np.argmax(similarities)
                    source_chunk_id = self.retriever.chunks[best_idx].get('chunk_id')
                
                results.append({
                    "id": str(uuid.uuid4()),
                    "instruction": instruction,
                    "input": "",
                    "output": output,
                    "metadata": {
                        "source_type": "expansion",
                        "source_chunk_id": source_chunk_id,
                        "task_type": "self_instruct",
                        "teacher_model": self.client.model,
                        "verified": False,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    }
                })
                
        return results
