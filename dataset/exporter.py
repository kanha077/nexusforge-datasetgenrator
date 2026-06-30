import json
import os
from typing import List
from dataset.schema import Sample

class Exporter:
    def __init__(self, output_dir: str = "datasets/exported"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def export_alpaca(self, samples: List[Sample], filename: str = "alpaca.json"):
        path = os.path.join(self.output_dir, filename)
        alpaca_format = []
        for sample in samples:
            alpaca_format.append({
                "instruction": sample.instruction,
                "input": sample.input,
                "output": sample.output
            })
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(alpaca_format, f, indent=2, ensure_ascii=False)

    def export_sharegpt(self, samples: List[Sample], filename: str = "sharegpt.jsonl"):
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            for sample in samples:
                sharegpt_format = {
                    "conversations": [
                        {"from": "human", "value": sample.instruction + ("\n" + sample.input if sample.input else "")},
                        {"from": "gpt", "value": sample.output}
                    ]
                }
                f.write(json.dumps(sharegpt_format, ensure_ascii=False) + '\n')
                
    def export_jsonl(self, samples: List[Sample], filename: str = "dataset.jsonl"):
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
