# 🌌 NexusForge

> A local synthetic dataset generation pipeline explicitly designed for fine-tuning highly stylized, in-voice chat personas.

NexusForge takes a small seed corpus (like your personal notes, journals, or training philosophies) and amplifies it into a high-quality, deduped, and rigorously verified dataset. It uses local LLMs via [Ollama](https://ollama.ai/) to generate both **grounded** samples directly tied to your seed data, and **expansion** (self-instruct) samples to widen the domain—all while strictly enforcing your desired persona.

---

## ✨ Features

- 🏗️ **Dual Generation Engine:** 
  - **Grounded Generation:** Extracts knowledge directly from `seed_data/raw/` to create factual QA, summarizations, instruction-following, and scenario applications.
  - **Expansion Generation:** Uses few-shot self-instruction to generate novel scenarios while strictly anchoring to your defined persona's voice and topic domains.
- 🎭 **Strict Persona Enforcement:** 
  - Uses cosine similarity against embedded persona descriptions to ensure generated outputs don't drift.
  - Hard keyword and topic gating prevents freewheeling hallucinations (e.g., stopping your fitness bro persona from giving WordPress tutorials).
- ⚖️ **Advanced Scoring & Quality Control:**
  - Evaluates every sample using a **Composite Score** (Persona Voice + Structural Penalty + Brevity Alignment).
  - Actively penalizes corporate fluff, generic listicles (`"Here's how:"`, numbered markdown lists), and verbose padding.
- 🛡️ **Robust Deduplication & Verification:**
  - Two-stage deduplication: Exact string fallback + Semantic embedding checks (`sentence_transformers`).
  - LLM-as-a-Judge verifier steps to explicitly grade topic adherence and voice matching.
- 📦 **Flexible Exports:** Seamlessly outputs to `Alpaca`, `ShareGPT`, and `JSONL` formats for immediate fine-tuning.

---

## 📂 Directory Structure

```text
nexusforge/
├── configs/              # Configuration files (generation params, prompts, project settings)
├── dataset/              # Assembly and exporter logic for final datasets
├── datasets/             # Output directory for exported .jsonl, alpaca, etc.
├── generators/           # Core LLM generation logic (Grounded, Expansion, Retriever)
├── inference/            # Ollama client and connection managers
├── knowledge/            # Text chunking and cleaning utilities
├── pipeline/             # Orchestrator, progress tracking, and checkpointing
├── prompts/              # Prompt templates (grounded, expansion, verification)
├── quality/              # Scorers, Verifiers, Dedupers, and Heuristic filters
└── seed_data/            
    └── raw/              # 📥 Drop your .txt seed files here!
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.9+**
- **Ollama** installed and running locally with your desired teacher model.
- **Python Dependencies:** `sentence_transformers`, `numpy`, `pyyaml`, `requests` (ensure these are installed in your environment).

### 2. Setup your Seed Corpus
Drop a few text files into `seed_data/raw/`. Even 3-4 small files (e.g., `01.txt`, `02.txt`) are enough to get started! NexusForge is heavily optimized to extract maximum value from extremely small, constrained datasets.

### 3. Configure the Persona
Edit `configs/project.yaml` to define your `persona`. Be specific about tone, voice, and constraints (e.g., *"A witty college student into fitness and sports. Sarcastic, dry, uses short sentences, absolutely no corporate fluff."*).

### 4. Adjust Generation Limits
Tweak `configs/generation.yaml`:
- `samples_per_chunk`: How many grounded variations to squeeze from a single text chunk.
- `expansion_ratio`: The percentage of the final dataset allowed to be "expansion" (self-instruct) samples (e.g., `0.2` = 20%).
- `min_quality_score`: Threshold for rejecting off-voice or corporate-sounding outputs.

### 5. Run the Pipeline
Execute the orchestrator to begin chunking, generation, and verification:
```bash
python -m pipeline.orchestrator
```
*(You may need to write a quick `main.py` entrypoint if you haven't already!)*

---

## 🧠 How the Quality Pipeline Works

1. **Generation:** The LLM generates a raw instruction/output pair.
2. **Topic Filter:** (Expansion only) Hard keyword check and semantic overlap check against seed data to prevent drift.
3. **LLM Verification:** `quality/verifier.py` checks if the output explicitly grounds to the text, matches the persona voice, and fits the topic domain.
4. **Scoring:** `quality/scorer.py` evaluates the output against brevity rules and penalizes robotic formatting. If `score < min_quality_score`, it is binned.
5. **Deduplication:** `quality/dedup.py` checks for exact matches and semantic similarity > `0.90` (default) against previously accepted samples.
6. **Assembly:** The sample is formatted and saved!

---

## 🛠️ Modifying Task Types

Want more variety? You can add new grounded tasks!
1. Add a new `.txt` prompt template to `prompts/grounded/` (e.g., `scenario_application.txt`).
2. Register it in `configs/prompts.yaml` under the `grounded:` list.
3. The orchestrator will automatically pick it up for the next run.

---

## 📜 License
MIT License - Have fun building your specialized personas! 🎉
