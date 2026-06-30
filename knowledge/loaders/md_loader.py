import os

def load_md(file_path: str) -> str:
    # Basic loader for now, could use markdown library to strip formatting
    # but for LLMs, raw markdown is often better anyway.
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def get_metadata(file_path: str) -> dict:
    return {
        "file_name": os.path.basename(file_path),
        "source_type": "md",
        "file_path": file_path
    }
