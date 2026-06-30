import os
import yaml
from typing import List, Dict, Any

from inference.client import OllamaClient
from inference.ollama_manager import OllamaManager
from generators.grounded_generator import GroundedGenerator
from generators.expansion_generator import ExpansionGenerator
from generators.retriever import Retriever
from quality.verifier import Verifier
from quality.heuristic_filter import HeuristicFilter
from quality.dedup import Deduplicator
from dataset.assembler import Assembler
from dataset.exporter import Exporter
from pipeline.checkpoint import CheckpointManager
from pipeline.progress import ProgressTracker
from knowledge.chunker import chunk_text
from knowledge.cleaner import clean_text
from quality.scorer import Scorer

class Orchestrator:
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        with open(os.path.join(config_dir, "project.yaml"), 'r') as f:
            self.project_config = yaml.safe_load(f)
        with open(os.path.join(config_dir, "generation.yaml"), 'r') as f:
            self.gen_config = yaml.safe_load(f)
        with open(os.path.join(config_dir, "prompts.yaml"), 'r') as f:
            self.prompts_config = yaml.safe_load(f)
            
        self.target_size = self.project_config.get("dataset_target_size", 5000)
        
        # Load prompt texts
        self.grounded_prompts = {}
        for item in self.prompts_config.get("grounded", []):
            path = os.path.join("prompts", "grounded", item["file"])
            if os.path.exists(path):
                with open(path, 'r') as f:
                    self.grounded_prompts[item["task_type"]] = f.read()
                    
        expansion_item = self.prompts_config.get("expansion", [])[0]
        with open(os.path.join("prompts", "expansion", expansion_item["file"]), 'r') as f:
            self.expansion_prompt = f.read()
            
        verification_item = self.prompts_config.get("verification", [])[0]
        with open(os.path.join("prompts", "verification", verification_item["file"]), 'r') as f:
            self.verification_prompt = f.read()

    def load_documents(self) -> List[Dict[str, Any]]:
        # A simple loader for TXT files in seed_data/raw
        chunks = []
        raw_dir = "seed_data/raw"
        if not os.path.exists(raw_dir):
            return chunks
            
        for filename in os.listdir(raw_dir):
            if filename.endswith(".txt"):
                path = os.path.join(raw_dir, filename)
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                cleaned_text = clean_text(text)
                file_chunks = chunk_text(cleaned_text, {"file_id": filename})
                chunks.extend(file_chunks)
        return chunks

    def run(self):
        # 1. Health check
        manager = OllamaManager(os.path.join(self.config_dir, "models.yaml"))
        if not manager.check_health():
            print("Failed to connect to Ollama. Exiting.")
            return

        # 2. Setup components
        client = OllamaClient(os.path.join(self.config_dir, "models.yaml"))
        retriever = Retriever()
        retriever._lazy_load()
        if retriever.model is None:
            raise RuntimeError("Embedding model failed to load. Halting to prevent duplicate bleed and unconstrained expansion.")
        
        scorer = Scorer(retriever, self.project_config.get("persona", ""))
        min_quality_score = self.gen_config.get("generation", {}).get("min_quality_score", 0.5)

        grounded_gen = GroundedGenerator(client, self.grounded_prompts)
        expansion_gen = ExpansionGenerator(client, self.expansion_prompt, retriever)
        
        verifier = Verifier(client, self.verification_prompt)
        dedup = Deduplicator(retriever, similarity_threshold=self.gen_config.get("diversity_threshold", 0.85))
        assembler = Assembler()
        exporter = Exporter()
        
        checkpoint = CheckpointManager()
        progress = ProgressTracker(self.target_size)
        progress.update(checkpoint.state["exported_samples"])

        # 3. Load & chunk knowledge
        chunks = self.load_documents()
        retriever.add_chunks(chunks)
        
        if not chunks:
            print("No source documents found in seed_data/raw. Please add some .txt files.")
            return

        accepted_samples = []

        # 4. Main Generation Loop
        expansion_ratio = self.gen_config.get("generation", {}).get("expansion_ratio", 0.5)
        grounded_target = int(self.target_size * (1.0 - expansion_ratio))

        for chunk in chunks:
            if progress.current_samples >= grounded_target:
                break
                
            if checkpoint.is_chunk_processed(chunk['chunk_id']):
                continue
                
            # Grounded generation
            for task_type in self.grounded_prompts.keys():
                raw_sample = grounded_gen.generate(chunk, task_type)
                if raw_sample and HeuristicFilter.is_valid(raw_sample):
                    if verifier.verify_grounding(raw_sample, chunk['text']):
                        raw_sample['metadata']['verified'] = True
                        if not dedup.is_duplicate(raw_sample):
                            score = scorer.score(raw_sample)
                            if score >= min_quality_score:
                                raw_sample.setdefault('metadata', {})['quality_score'] = score
                                dedup.add_sample(raw_sample)
                                accepted_samples.append(assembler.assemble(raw_sample))
                                progress.update(1)
                            
                            if progress.current_samples % 70 == 0 and progress.current_samples > 0:
                                print(f"\n[PAUSE] Generated {progress.current_samples} samples. Saving current progress...")
                                formats = self.project_config.get("export_formats", ["jsonl"])
                                if "alpaca" in formats:
                                    exporter.export_alpaca(accepted_samples)
                                if "sharegpt" in formats:
                                    exporter.export_sharegpt(accepted_samples)
                                if "jsonl" in formats:
                                    exporter.export_jsonl(accepted_samples)
                                checkpoint.update_exported_samples(len(accepted_samples))
                                
                                ans = input("Check the outputs/exported folder. Continue generating? (y/n): ")
                                if ans.lower().strip() != 'y':
                                    print("Stopping generation.")
                                    return
                            
            checkpoint.mark_chunk_processed(chunk['chunk_id'])

        import random
        # 4.5. Phase 2: Infinite Expansion Generation
        max_expansion_samples = int(self.target_size * self.gen_config.get("generation", {}).get("expansion_ratio", 0.2))
        expansion_accepted = 0
        
        if len(accepted_samples) >= 3:
            print("\nStarting continuous expansion generation...")
            while expansion_accepted < max_expansion_samples:
                # Prefer grounded samples for few-shot to prevent drift
                grounded_samples = [s for s in accepted_samples if s.get('metadata', {}).get('source_type') == 'grounded']
                if len(grounded_samples) >= 2:
                    selected = random.sample(grounded_samples, 2)
                    others = [s for s in accepted_samples if s not in selected]
                    if others:
                        selected.append(random.choice(others))
                else:
                    selected = random.sample(accepted_samples, min(3, len(accepted_samples)))
                    
                examples = [s.to_dict() if hasattr(s, 'to_dict') else s for s in selected]
                persona = self.project_config.get("persona", "")
                
                # Check topic and voice in verifier
                raw_expansions = expansion_gen.generate(examples, num_samples=3, persona=persona)
                
                if not raw_expansions:
                    continue
                    
                for exp_sample in raw_expansions:
                    if progress.current_samples >= self.target_size:
                        break
                    if HeuristicFilter.is_valid(exp_sample):
                        if verifier.verify_expansion(exp_sample, persona):
                            exp_sample['metadata']['verified'] = True
                            if not dedup.is_duplicate(exp_sample):
                                score = scorer.score(exp_sample)
                                if score >= min_quality_score:
                                    exp_sample.setdefault('metadata', {})['quality_score'] = score
                                    dedup.add_sample(exp_sample)
                                    accepted_samples.append(assembler.assemble(exp_sample))
                                    progress.update(1)
                                    expansion_accepted += 1
                            
                            if progress.current_samples % 70 == 0 and progress.current_samples > 0:
                                print(f"\n[PAUSE] Generated {progress.current_samples} samples. Saving current progress...")
                                formats = self.project_config.get("export_formats", ["jsonl"])
                                if "alpaca" in formats:
                                    exporter.export_alpaca(accepted_samples)
                                if "sharegpt" in formats:
                                    exporter.export_sharegpt(accepted_samples)
                                if "jsonl" in formats:
                                    exporter.export_jsonl(accepted_samples)
                                checkpoint.update_exported_samples(len(accepted_samples))
                                
                                ans = input("Check the outputs/exported folder. Continue generating? (y/n): ")
                                if ans.lower().strip() != 'y':
                                    print("Stopping generation.")
                                    return

        # 5. Export
        print(f"\nFinal tally: {len(accepted_samples)} total samples ({len(accepted_samples) - expansion_accepted} grounded, {expansion_accepted} expansion).")
        if progress.current_samples < self.target_size:
            print(f"Run stopped short of target_size ({self.target_size}) due to expansion_ratio cap and limited seed data.")
        print(f"\nExporting {len(accepted_samples)} samples...")
        formats = self.project_config.get("export_formats", ["jsonl"])
        if "alpaca" in formats:
            exporter.export_alpaca(accepted_samples)
        if "sharegpt" in formats:
            exporter.export_sharegpt(accepted_samples)
        if "jsonl" in formats:
            exporter.export_jsonl(accepted_samples)
            
        checkpoint.update_exported_samples(len(accepted_samples))
        progress.complete()
