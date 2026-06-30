import os

def load_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def get_metadata(file_path: str) -> dict:
    return {
        "file_name": os.path.basename(file_path),
        "source_type": "txt",
        "file_path": file_path
    }
