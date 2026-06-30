import uuid
import time
from typing import Dict, Any, Optional
from inference.client import OllamaClient

class GroundedGenerator:
    def __init__(self, client: OllamaClient, prompts: Dict[str, str]):
        self.client = client
        self.prompts = prompts

    def generate(self, chunk: Dict[str, Any], task_type: str) -> Optional[Dict[str, Any]]:
        prompt_template = self.prompts.get(task_type)
        if not prompt_template:
            raise ValueError(f"No prompt template found for task type: {task_type}")
            
        prompt = prompt_template.format(text=chunk['text'])
        response = self.client.generate(prompt=prompt, max_tokens=1024)
        
        # Parse output
        instruction = ""
        output = ""
        lines = response.split('\n')
        parsing_instruction = False
        parsing_output = False
        
        for line in lines:
            if line.startswith("Instruction:"):
                instruction = line[len("Instruction:"):].strip()
                parsing_instruction = True
                parsing_output = False
            elif line.startswith("Output:"):
                output = line[len("Output:"):].strip()
                parsing_instruction = False
                parsing_output = True
            elif parsing_instruction:
                instruction += "\n" + line
            elif parsing_output:
                output += "\n" + line
                
        instruction = instruction.strip()
        output = output.strip()
        
        if not instruction or not output:
            return None
            
        return {
            "id": str(uuid.uuid4()),
            "instruction": instruction,
            "input": "",
            "output": output,
            "metadata": {
                "source_type": "grounded",
                "source_chunk_id": chunk.get('chunk_id', 'unknown'),
                "task_type": task_type,
                "teacher_model": self.client.model,
                "verified": False,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        }
