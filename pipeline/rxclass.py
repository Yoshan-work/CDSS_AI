"""
WHO/ISO ATC & NLM RxClass Lookup Module.
Resolves a normalized drug (generic name + RxCUI) to its official ATC code,
ATC Level-4 chemical/therapeutic subgroup, pharmacologic class, and allergy group.
Queries NLM RxClass REST API on-demand if a drug is not in the local table.
"""

import json
from typing import Any, Dict, Optional
import requests

from config import ATC_CLASSES_PATH, RXCLASS_BASE_URL
from engine.trace_logger import iso_logger

_ATC_CACHE: Optional[Dict[str, Any]] = None


def load_atc_database() -> Dict[str, Any]:
    """Load local WHO/ISO ATC classification & allergy cross-reactivity database."""
    global _ATC_CACHE
    if _ATC_CACHE is not None:
        return _ATC_CACHE
    if not ATC_CLASSES_PATH.exists():
        _ATC_CACHE = {"allergy_cross_reactivity_matrix": {}, "drugs": {}}
        return _ATC_CACHE
    with open(ATC_CLASSES_PATH, "r", encoding="utf-8") as f:
        _ATC_CACHE = json.load(f)
    return _ATC_CACHE


def resolve_drug_atc_and_class(
    generic_name: str, rxcui: str = "UNKNOWN", timeout: int = 6
) -> Dict[str, Any]:
    """
    Resolve a drug's WHO/ISO ATC code, therapeutic class, and allergy group.
    1. Checks local curated `Data/atc_classes.json` first (0ms offline).
    2. If not found and RxCUI is valid, queries NLM RxClass REST API (`relaSource=ATC`).
    """
    db = load_atc_database()
    drugs_map = db.get("drugs", {})
    key = generic_name.strip().lower()

    if key in drugs_map:
        entry = dict(drugs_map[key])
        entry["source"] = "WHO/ISO ATC Local Reference"
        return entry

    # Check if combination drug contains a known ingredient (e.g., amoxicillin / clavulanate)
    for known_drug, info in drugs_map.items():
        if known_drug in key:
            entry = dict(info)
            entry["source"] = f"WHO/ISO ATC Ingredient Match ({known_drug})"
            return entry

    # Fallback: Query NLM RxClass REST API for ATC classification
    if rxcui and rxcui != "UNKNOWN":
        try:
            resp = requests.get(
                f"{RXCLASS_BASE_URL}/class/byRxcui.json",
                params={"rxcui": rxcui, "relaSource": "ATC"},
                timeout=timeout,
            )
            if resp.ok:
                rxclass_list = (
                    resp.json()
                    .get("rxclassDrugInfoList", {})
                    .get("rxclassDrugInfo", [])
                )
                if rxclass_list:
                    concept = rxclass_list[0].get("rxclassMinConceptItem", {})
                    class_id = str(concept.get("classId", "UNKNOWN"))
                    class_name = str(concept.get("className", "Unclassified"))
                    return {
                        "atc_code": class_id,
                        "atc_level4": class_id[:5] if len(class_id) >= 5 else class_id,
                        "therapeutic_class": class_name,
                        "allergy_group": key,
                        "source": "NLM RxClass REST API (ATC)",
                    }
        except Exception as exc:
            iso_logger.debug(f"RxClass live lookup skipped for {generic_name} ({rxcui}): {exc}")

    return {
        "atc_code": "UNCLASSIFIED",
        "atc_level4": "UNCLASSIFIED",
        "therapeutic_class": "Unclassified",
        "allergy_group": key,
        "source": "Unclassified",
    }

