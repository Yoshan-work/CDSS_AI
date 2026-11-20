import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
CHUNKS_DIR = DATA_DIR / "chunks"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# File paths
CHUNKS_JSONL_PATH = CHUNKS_DIR / "knowledge_chunks.jsonl"
VECTOR_DB_PATH = DATA_DIR / "vector_store.sqlite"
FORMULARY_SEED_PATH = DATA_DIR / "formulary_seed.json"
WHO_EML_PATH = DATA_DIR / "who_eml_core.json"
EVALUATION_HISTORY_PATH = DATA_DIR / "evaluation_history.jsonl"

# External Free Clinical APIs
RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"
OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
OPENFDA_API_KEY = os.environ.get("OPENFDA_API_KEY", "")  # Optional free key to raise rate limit
DAILYMED_BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

# Local Ollama Qwen LLM Configuration
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_FALLBACK_MODELS = ["qwen3:8b", "qwen3-vl:4b", "qwen2.5:7b", "qwen:latest"]

# Target Label Sections for RAG Chunks
TARGET_LABEL_SECTIONS = [
    "indications_and_usage",
    "contraindications",
    "drug_interactions",
    "dosage_and_administration",
    "warnings",
    "pregnancy",
]

