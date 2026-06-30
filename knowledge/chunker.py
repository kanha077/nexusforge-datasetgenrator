import tiktoken
from typing import List, Dict, Any

def get_tokenizer(model_name: str = "gpt-3.5-turbo"):
    # Using tiktoken cl100k_base as a fast generic tokenizer approximation for GGUF models 
    # unless a specific HF tokenizer is loaded.
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")

def chunk_text(text: str, source_metadata: Dict[str, Any], chunk_size: int = 512, chunk_overlap: int = 64) -> List[Dict[str, Any]]:
    """
    Chunks text into context-sized pieces with overlap.
    Returns a list of dicts with 'text' and 'metadata'.
    """
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text)
    
    chunks = []
    start_idx = 0
    chunk_index = 0
    
    while start_idx < len(tokens):
        end_idx = min(start_idx + chunk_size, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens)
        
        chunks.append({
            "chunk_id": f"{source_metadata.get('file_id', 'doc')}_{chunk_index}",
            "text": chunk_text,
            "metadata": {
                **source_metadata,
                "chunk_index": chunk_index,
                "start_token": start_idx,
                "end_token": end_idx
            }
        })
        
        start_idx += (chunk_size - chunk_overlap)
        chunk_index += 1
        
    return chunks
