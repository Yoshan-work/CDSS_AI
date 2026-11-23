"""
ISO-Compliant Central Configuration Module for Clinical Decision Support System (CDSS).
Standards Referenced:
- ISO 8601:2019 (Date and time format with timezone offset)
- ISO/IEC 25010 (System and software quality models — Reliability, Traceability)
- WHO/ISO ATC Classification (Anatomical Therapeutic Chemical code system)
- HL7 SPL / LOINC Section Codes & NLM RxNorm RxCUI
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv(env_path: Path) -> None:
    """Load key=value pairs from .env into os.environ without overriding existing env vars."""
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


_load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "Data"
CHUNKS_DIR = DATA_DIR / "chunks"
LOGS_DIR = BASE_DIR / "logs"

# Server Configuration
CDSS_HOST = os.environ.get("CDSS_HOST", "127.0.0.1")
CDSS_PORT = int(os.environ.get("CDSS_PORT", "8000"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Knowledge Base & Structured Rule File Paths
CHUNKS_JSONL_PATH = CHUNKS_DIR / "knowledge_chunks.jsonl"
VECTOR_DB_PATH = DATA_DIR / "vector_store.sqlite"
DDINTER_DB_PATH = DATA_DIR / "ddinter_db.sqlite"
DDINTER_SEED_PATH = DATA_DIR / "ddinter_core_seed.json"
ATC_CLASSES_PATH = DATA_DIR / "atc_classes.json"
DOSE_RULES_PATH = DATA_DIR / "dose_rules.json"
FORMULARY_SEED_PATH = DATA_DIR / "formulary_seed.json"
WHO_EML_PATH = DATA_DIR / "who_eml_core.json"

# Audit & Trace Log File Paths (ISO 8601 Traceability)
EVALUATION_HISTORY_PATH = DATA_DIR / "evaluation_history.jsonl"
SYSTEM_TRACE_JSONL_PATH = LOGS_DIR / "cdss_system_trace.jsonl"
SYSTEM_LOG_FILE_PATH = LOGS_DIR / "cdss_system.log"

# External Clinical APIs
RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"
RXCLASS_BASE_URL = "https://rxnav.nlm.nih.gov/REST/rxclass"
OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
OPENFDA_API_KEY = os.environ.get("OPENFDA_API_KEY", "")
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
