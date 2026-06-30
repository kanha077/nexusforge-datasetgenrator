from typing import Dict, Any
from dataset.schema import Sample, SampleMetadata

class Assembler:
    @staticmethod
    def assemble(raw_sample: Dict[str, Any]) -> Sample:
        meta = raw_sample.get('metadata', {})
        metadata = SampleMetadata(
            source_type=meta.get('source_type', 'unknown'),
            source_chunk_id=meta.get('source_chunk_id'),
            task_type=meta.get('task_type', 'unknown'),
            teacher_model=meta.get('teacher_model', 'unknown'),
            verified=meta.get('verified', False),
            quality_score=meta.get('quality_score', 1.0),
            created_at=meta.get('created_at', '')
        )
        return Sample(
            id=raw_sample['id'],
            instruction=raw_sample['instruction'],
            input=raw_sample.get('input', ''),
            output=raw_sample['output'],
            metadata=metadata
        )
