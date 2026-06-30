import re
import numpy as np
from typing import Dict, Any
from generators.retriever import Retriever

PERSONA_VOICE_WEIGHT = 0.5
STRUCTURE_PENALTY_WEIGHT = 0.4
BREVITY_ALIGNMENT_WEIGHT = 0.3

class Scorer:
    def __init__(self, retriever: Retriever, persona: str):
        self.retriever = retriever
        self.persona = persona
        self.retriever._lazy_load()
        if self.retriever.model:
            self.persona_embedding = self.retriever.model.encode([persona], convert_to_numpy=True)[0]
        else:
            self.persona_embedding = None

    def score(self, sample: Dict[str, Any]) -> float:
        output_text = sample.get('output', '')
        
        # 1. Persona Voice Score
        persona_score = 0.0
        if self.persona_embedding is not None and self.retriever.model:
            output_emb = self.retriever.model.encode([output_text], convert_to_numpy=True)[0]
            sim = np.dot(self.persona_embedding, output_emb) / (
                np.linalg.norm(self.persona_embedding) * np.linalg.norm(output_emb)
            )
            persona_score = float(max(0.0, min(1.0, sim)))
            
        # 2. Structure Penalty
        structure_penalty = 0.0
        if re.search(r'^\d+\.\s+\*\*', output_text, re.MULTILINE):
            structure_penalty = 1.0
        elif len(re.findall(r'\*\*[^\*]+\*\*', output_text)) >= 3:
            structure_penalty = 1.0
        elif re.search(r"here's how|here are some tips", output_text, re.IGNORECASE):
            structure_penalty = 1.0
            
        # 3. Brevity Alignment
        output_len = len(output_text)
        if output_len < 20:
            brevity_score = 0.2
        elif output_len <= 300:
            brevity_score = 1.0
        elif output_len <= 500:
            brevity_score = 0.7
        else:
            brevity_score = max(0.0, 1.0 - ((output_len - 500) / 1000.0))
            
        final_score = (
            PERSONA_VOICE_WEIGHT * persona_score
            + BREVITY_ALIGNMENT_WEIGHT * brevity_score
            - STRUCTURE_PENALTY_WEIGHT * structure_penalty
        )
        return min(1.0, max(0.0, final_score))

