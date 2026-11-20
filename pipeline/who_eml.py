import json
import logging
from datetime import date
from pathlib import Path
from typing import Dict, Any, List, Optional
import pdfplumber

from config import WHO_EML_PATH

logger = logging.getLogger(__name__)


def load_who_eml_reference() -> Dict[str, Any]:
    """Load the structured WHO Essential Medicines List reference JSON."""
    if not WHO_EML_PATH.exists():
        return {"medicines": {}}
    try:
        with open(WHO_EML_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"Failed to load WHO EML reference: {exc}")
        return {"medicines": {}}


def get_who_eml_info(generic_name: str) -> Dict[str, Any]:
    """
    Check whether a generic drug is in the WHO Essential Medicines reference list
    and return its EML status, recommended indications, typical dose, and WHO notes.
    """
    data = load_who_eml_reference()
    meds = data.get("medicines", {})
    key = generic_name.strip().lower()

    if key in meds:
        entry = dict(meds[key])
        entry["drug"] = key
        entry["in_who_eml"] = "Not on WHO EML" not in entry.get("eml_status", "")
        return entry

    # Also check inn_alias matches (e.g., paracetamol <-> acetaminophen)
    for med_key, med_val in meds.items():
        if med_val.get("inn_alias", "").lower() == key:
            entry = dict(med_val)
            entry["drug"] = med_key
            entry["in_who_eml"] = True
            return entry

    return {
        "drug": key,
        "in_who_eml": False,
        "eml_status": "Not indexed in starter WHO EML core subset (Check full WHO EML / NMRA Formulary)",
        "section": "General Formulary",
        "recommended_indications": "See openFDA / DailyMed label indications",
        "who_notes": "",
        "typical_adult_dose": "",
    }


def build_who_eml_chunk(generic_name: str, rxcui: str = "UNKNOWN") -> Optional[Dict[str, Any]]:
    """Convert a WHO EML entry into a RAG knowledge chunk for the Vector DB."""
    info = get_who_eml_info(generic_name)
    if not info.get("who_notes") and not info.get("typical_adult_dose"):
        return None

    text = (
        f"WHO Essential Medicines List Status: {info.get('eml_status')} ({info.get('section')}). "
        f"Recommended Indications: {info.get('recommended_indications')}. "
        f"Reference Adult Dose: {info.get('typical_adult_dose')}. "
        f"Clinical & Safety Notes: {info.get('who_notes')}"
    )
    return {
        "drug": generic_name.lower(),
        "rxcui": str(rxcui),
        "section": "who_eml_reference",
        "text": text,
        "source": "WHO Essential Medicines List (23rd EML Reference)",
        "label_version_date": "2023-07",
        "retrieved_on": date.today().isoformat(),
    }


def parse_who_eml_pdf(pdf_path: str | Path, target_drugs: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Utility to parse an official WHO Essential Medicines List PDF (or Sri Lanka National Formulary PDF)
    using pdfplumber and convert matched pages into knowledge chunks.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    today_str = date.today().isoformat()
    chunks: List[Dict[str, Any]] = []
    target_set = {d.lower() for d in target_drugs} if target_drugs else set()

    with pdfplumber.open(path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue
            lower_text = page_text.lower()
            if target_set:
                matched_drugs = [d for d in target_set if d in lower_text]
                for drug in matched_drugs:
                    chunks.append(
                        {
                            "drug": drug,
                            "rxcui": "WHO_PDF",
                            "section": "who_eml_pdf",
                            "text": page_text[:3000],
                            "source": f"WHO EML PDF ({path.name} p.{page_idx})",
                            "label_version_date": "PDF",
                            "retrieved_on": today_str,
                        }
                    )
            else:
                chunks.append(
                    {
                        "drug": "general_guideline",
                        "rxcui": "WHO_PDF",
                        "section": "who_eml_pdf",
                        "text": page_text[:3000],
                        "source": f"WHO EML PDF ({path.name} p.{page_idx})",
                        "label_version_date": "PDF",
                        "retrieved_on": today_str,
                    }
                )
    return chunks

