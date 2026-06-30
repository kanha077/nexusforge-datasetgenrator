import numpy as np
from typing import List, Dict, Any

class Retriever:
    def __init__(self, model_name="nomic-ai/nomic-embed-text-v1.5", device="cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.embeddings = []
        self.chunks = []
        
    def _lazy_load(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name, device=self.device, trust_remote_code=True)
            except ImportError as e:
                print(f"Warning: Failed to load embedding model. Error: {e}")

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        self._lazy_load()
        if not self.model: return
        
        texts = [chunk['text'] for chunk in chunks]
        new_embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        if len(self.embeddings) == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        self.chunks.extend(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        self._lazy_load()
        if not self.model or len(self.chunks) == 0: return []
        
        query_emb = self.model.encode([query], convert_to_numpy=True)[0]
        # Cosine similarity
        similarities = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.chunks[i] for i in top_indices]
