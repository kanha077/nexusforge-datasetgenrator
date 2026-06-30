from typing import Dict, Any
from inference.client import OllamaClient

class Verifier:
    def __init__(self, client: OllamaClient, prompt_template: str):
        self.client = client
        self.prompt_template = prompt_template

    def verify_grounding(self, sample: Dict[str, Any], source_text: str) -> bool:
        prompt = self.prompt_template.format(
            text=source_text,
            instruction=sample['instruction'],
            output=sample['output']
        )
        response = self.client.generate(prompt=prompt, max_tokens=10).strip().lower()
        return "yes" in response

    def verify_expansion(self, sample: Dict[str, Any], persona: str) -> bool:
        topic_prompt = f"Does the following instruction and output fall strictly into one of these topics: fitness training/recovery/nutrition, sports stats/debates, deadlines/procrastination, studying/active recall, group projects, sleep vs. assignments, decision fatigue, discipline/motivation in a school context?\nInstruction: {sample['instruction']}\nOutput: {sample['output']}\nAnswer yes or no."
        topic_resp = self.client.generate(prompt=topic_prompt, max_tokens=10).strip().lower()
        if "yes" not in topic_resp:
            return False
            
        voice_prompt = f"Does the following output strictly match this persona's voice (witty, conversational, sarcastic, no corporate fluff, short sentences)?\nPersona: {persona}\nOutput: {sample['output']}\nAnswer yes or no."
        voice_resp = self.client.generate(prompt=voice_prompt, max_tokens=10).strip().lower()
        return "yes" in voice_resp
