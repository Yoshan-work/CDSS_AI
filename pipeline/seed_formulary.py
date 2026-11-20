import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

# Allow running directly as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FORMULARY_SEED_PATH
from pipeline.vector_store import CDSSVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Priority starter subset (seed immediately on first launch or via CLI)
PRIORITY_STARTER_DRUGS = [
    "ibuprofen",
    "acetaminophen",
    "aspirin",
    "diclofenac",
    "warfarin",
    "clopidogrel",
    "amoxicillin",
    "azithromycin",
    "ciprofloxacin",
    "metronidazole",
    "metformin",
    "amlodipine",
    "losartan",
    "enalapril",
    "furosemide",
    "spironolactone",
    "atorvastatin",
    "simvastatin",
    "omeprazole",
    "prednisolone",
]


def seed_formulary_drugs(drug_list: List[str] = None, limit: int = 20) -> Dict[str, Any]:
    """
    Normalize each target drug with RxNorm -> pull openFDA/DailyMed sections ->
    attach WHO EML reference -> embed & store in Vector DB.
    """
    if drug_list is None:
        if FORMULARY_SEED_PATH.exists():
            with open(FORMULARY_SEED_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                drug_list = data.get("drugs", PRIORITY_STARTER_DRUGS)[:limit]
        else:
            drug_list = PRIORITY_STARTER_DRUGS[:limit]

    store = CDSSVectorStore()
    seeded_summary = []

    for idx, drug in enumerate(drug_list, start=1):
        logger.info(f"[{idx}/{len(drug_list)}] Indexing '{drug}' via RxNorm + openFDA/DailyMed + WHO EML...")
        res = store.ensure_drug_indexed(drug)
        norm = res["normalization"]
        chunks = res["chunks"]
        seeded_summary.append(
            {
                "entered": drug,
                "generic_name": norm["generic_name"],
                "rxcui": norm["rxcui"],
                "chunks_stored": len(chunks),
                "sections": [c["section"] for c in chunks],
            }
        )

    stats = store.get_stats()
    return {
        "processed_count": len(seeded_summary),
        "seeded_drugs": seeded_summary,
        "db_stats": stats,
    }


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    result = seed_formulary_drugs(PRIORITY_STARTER_DRUGS[:count], limit=count)
    print(
        f"\nSeeded {result['processed_count']} drugs. "
        f"Total DB Drugs: {result['db_stats']['total_drugs']}, "
        f"Total Chunks: {result['db_stats']['total_chunks']}"
    )

