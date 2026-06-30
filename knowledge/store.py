import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np

class ChunkStore:
    def __init__(self, store_path="cache/chunk_store.jsonl", embedding_model="nomic-ai/nomic-embed-text-v1.5", device="cpu"):
        self.store_path = store_path
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        self.chunks = []
        self.embeddings = None
        
        print(f"Loading embedding model {embedding_model} on {device}...")
        self.model = SentenceTransformer(embedding_model, device=device)
        self.model.trust_remote_code = True
        
        self.load_store()

    def load_store(self):
        if not os.path.exists(self.store_path):
            return
            
        with open(self.store_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.chunks.append(json.loads(line))
        
        if self.chunks:
            # Rebuild embeddings array from loaded chunks
            emb_list = [c.get("embedding") for c in self.chunks]
            if all(e is not None for e in emb_list):
                self.embeddings = np.array(emb_list, dtype=np.float32)

    def add_chunks(self, new_chunks):
        if not new_chunks:
            return
            
        texts = [c["text"] for c in new_chunks]
        print(f"Embedding {len(texts)} chunks...")
        # Use prefix for nomic-embed if needed, but standard encode is usually fine
        new_embs = self.model.encode(texts, normalize_embeddings=True)
        
        with open(self.store_path, 'a', encoding='utf-8') as f:
            for i, chunk in enumerate(new_chunks):
                chunk["embedding"] = new_embs[i].tolist()
                f.write(json.dumps(chunk) + '\n')
                self.chunks.append(chunk)
                
        if self.embeddings is None:
            self.embeddings = new_embs
        else:
            self.embeddings = np.vstack([self.embeddings, new_embs])

    def search(self, query: str, top_k: int = 3):
        if self.embeddings is None or len(self.chunks) == 0:
            return []
            
        query_emb = self.model.encode([query], normalize_embeddings=True)[0]
        similarities = np.dot(self.embeddings, query_emb)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(similarities[idx])
            })
        return results

    def get_random_chunks(self, k: int = 5):
        if not self.chunks:
            return []
        indices = np.random.choice(len(self.chunks), min(k, len(self.chunks)), replace=False)
        return [self.chunks[i] for i in indices]
