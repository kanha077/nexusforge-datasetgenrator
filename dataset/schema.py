from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class SampleMetadata:
    source_type: str
    source_chunk_id: Optional[str]
    task_type: str
    teacher_model: str
    verified: bool
    quality_score: float
    created_at: str

@dataclass
class Sample:
    id: str
    instruction: str
    input: str
    output: str
    metadata: SampleMetadata

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
