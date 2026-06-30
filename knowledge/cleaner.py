import re

def clean_text(text: str) -> str:
    """
    Strips boilerplate and normalizes whitespace/encoding.
    """
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Normalize unicode quotes and dashes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('–', '-').replace('—', '-')
    
    # Normalize whitespace: convert multiple spaces to a single space
    # Convert 3+ newlines to double newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()
