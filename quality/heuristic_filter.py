import re
from typing import Dict, Any

class HeuristicFilter:
    REFUSAL_TERMS = ["as an ai", "i cannot", "i am an ai", "i'm sorry", "i apologize", "as a language model"]
    
    @staticmethod
    def is_valid(sample: Dict[str, Any]) -> bool:
        instruction = sample.get('instruction', '').lower()
        output = sample.get('output', '').lower()
        
        # Length sanity
        if len(instruction) < 10 or len(output) < 10:
            return False
            
        # Refusal check
        for term in HeuristicFilter.REFUSAL_TERMS:
            if term in output:
                return False
                
        # Simple repetition check
        words = output.split()
        if len(words) > 10:
            for i in range(len(words) - 5):
                if words[i] == words[i+1] == words[i+2] == words[i+3] == words[i+4]:
                    return False
                    
        return True
