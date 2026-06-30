import numpy as np
from typing import List, Dict, Any
from generators.retriever import Retriever

class Deduplicator:
    def __init__(self, retriever: Retriever, similarity_threshold: float = 0.85):
        self.retriever = retriever
        self.similarity_threshold = similarity_threshold
        self.accepted_samples = []
        self.accepted_embeddings = []

    def is_duplicate(self, sample: Dict[str, Any]) -> bool:
        if not self.accepted_samples:
            return False
            
        # Fallback exact string match
        query_text = f"Instruction: {sample['instruction']}\nOutput: {sample['output']}".strip().lower()
        for accepted in self.accepted_samples:
            accepted_text = f"Instruction: {accepted['instruction']}\nOutput: {accepted['output']}".strip().lower()
            if query_text == accepted_text:
                return True
            
        self.retriever._lazy_load()
        if not self.retriever.model:
            return False
            
        text_to_embed = f"Instruction: {sample['instruction']}\nOutput: {sample['output']}"
        query_emb = self.retriever.model.encode([text_to_embed], convert_to_numpy=True)[0]
        
        similarities = np.dot(self.accepted_embeddings, query_emb) / (
            np.linalg.norm(self.accepted_embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        
        if np.max(similarities) >= self.similarity_threshold:
            return True
        return False
        
    def add_sample(self, sample: Dict[str, Any]):
        self.retriever._lazy_load()
        if not self.retriever.model:
            return
            
        text_to_embed = f"Instruction: {sample['instruction']}\nOutput: {sample['output']}"
        query_emb = self.retriever.model.encode([text_to_embed], convert_to_numpy=True)[0]
        
        if len(self.accepted_embeddings) == 0:
            self.accepted_embeddings = np.array([query_emb])
        else:
            self.accepted_embeddings = np.vstack([self.accepted_embeddings, query_emb])
            
        self.accepted_samples.append(sample)
