import re
import logging
import xml.etree.ElementTree as ET
from datetime import date
from typing import List, Dict, Any, Optional
import requests

from config import (
    OPENFDA_LABEL_URL,
    OPENFDA_API_KEY,
    DAILYMED_BASE_URL,
    TARGET_LABEL_SECTIONS,
)

logger = logging.getLogger(__name__)

# LOINC Section Codes in DailyMed SPL XML mapped to our 6 target sections
DAILYMED_LOINC_SECTION_MAP = {
    "34067-9": "indications_and_usage",
    "34070-3": "contraindications",
    "34073-7": "drug_interactions",
    "34068-7": "dosage_and_administration",
    "34071-1": "warnings",
    "43685-7": "warnings",          # Warnings and Precautions
    "34066-1": "warnings",          # Boxed Warning
    "42228-7": "pregnancy",         # Pregnancy
    "34080-2": "pregnancy",         # Nursing Mothers
}


def _clean_text(text: str, max_chars: int = 3500) -> str:
    """Clean excessive whitespace and truncate very long tables gracefully for RAG chunks."""
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(" ", 1)[0] + " ... [truncated]"
    return cleaned


def fetch_openfda_sections(
    generic_name: str, rxcui: Optional[str] = None, timeout: int = 10
) -> Dict[str, Any]:
    """
    Query openFDA Drug Label API (https://api.fda.gov/drug/label.json).
    Tries openfda.generic_name:"{generic_name}", openfda.rxcui:"{rxcui}", and openfda.substance_name.
    Returns dict of extracted sections + effective_time (label_version_date).
    """
    queries = [
        f'openfda.generic_name:"{generic_name}"',
        f'openfda.substance_name:"{generic_name.upper()}"',
    ]
    if rxcui and rxcui != "UNKNOWN":
        queries.insert(1, f'openfda.rxcui:"{rxcui}"')

    session = requests.Session()
    best_label: Optional[Dict[str, Any]] = None
    best_score = -1

    for q in queries:
        params: Dict[str, Any] = {"search": q, "limit": 3}
        if OPENFDA_API_KEY:
            params["api_key"] = OPENFDA_API_KEY
        try:
            resp = session.get(OPENFDA_LABEL_URL, params=params, timeout=timeout)
            if not resp.ok:
                continue
            results = resp.json().get("results", [])
            for item in results:
                # Prefer oral/systemic prescription labels that have drug_interactions & contraindications
                score = sum(
                    1
                    for sec in TARGET_LABEL_SECTIONS
                    if item.get(sec) or (sec == "warnings" and item.get("warnings_and_cautions"))
                )
                if item.get("drug_interactions"):
                    score += 2
                if item.get("contraindications"):
                    score += 2
                if score > best_score:
                    best_score = score
                    best_label = item
            if best_score >= 6:
                break
        except Exception as exc:
            logger.debug(f"openFDA query '{q}' failed: {exc}")

    if not best_label:
        return {"sections": {}, "label_version_date": "unknown", "source": "openFDA"}

    sections: Dict[str, str] = {}
    label_date = str(best_label.get("effective_time", "unknown"))

    for field in TARGET_LABEL_SECTIONS:
        raw_parts: List[str] = []
        if field == "warnings":
            for wf in ["boxed_warning", "warnings_and_cautions", "warnings"]:
                val = best_label.get(wf)
                if isinstance(val, list):
                    raw_parts.extend(val)
        elif field == "pregnancy":
            for pf in ["pregnancy", "teratogenic_effects", "nursing_mothers"]:
                val = best_label.get(pf)
                if isinstance(val, list):
                    raw_parts.extend(val)
        else:
            val = best_label.get(field)
            if isinstance(val, list):
                raw_parts.extend(val)

        combined = _clean_text(" ".join(raw_parts))
        if combined:
            sections[field] = combined

    return {
        "sections": sections,
        "label_version_date": label_date,
        "source": "openFDA",
    }


def fetch_dailymed_sections(generic_name: str, timeout: int = 12) -> Dict[str, Any]:
    """
    Fallback fetcher for DailyMed SPL XML:
    1. Search https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?drug_name={generic_name}
    2. Take setid from result, then fetch XML at .../spls/{setid}.xml
    3. Extract LOINC-coded clinical sections.
    """
    session = requests.Session()
    try:
        search_url = f"{DAILYMED_BASE_URL}/spls.json"
        resp = session.get(search_url, params={"drug_name": generic_name, "pagesize": 3}, timeout=timeout)
        if not resp.ok:
            return {"sections": {}, "label_version_date": "unknown", "source": "DailyMed"}

        data = resp.json().get("data", [])
        if not data:
            return {"sections": {}, "label_version_date": "unknown", "source": "DailyMed"}

        setid = data[0].get("setid")
        published_date = str(data[0].get("published_date", "unknown"))
        if not setid:
            return {"sections": {}, "label_version_date": "unknown", "source": "DailyMed"}

        xml_url = f"{DAILYMED_BASE_URL}/spls/{setid}.xml"
        xml_resp = session.get(xml_url, timeout=timeout)
        if not xml_resp.ok:
            return {"sections": {}, "label_version_date": published_date, "source": "DailyMed"}

        root = ET.fromstring(xml_resp.content)
        ns = {"hl7": "urn:hl7-org:v3"}

        extracted_lists: Dict[str, List[str]] = {sec: [] for sec in TARGET_LABEL_SECTIONS}

        for section_elem in root.findall(".//hl7:section", ns):
            code_elem = section_elem.find("hl7:code", ns)
            if code_elem is None:
                continue
            loinc_code = code_elem.attrib.get("code", "")
            target_field = DAILYMED_LOINC_SECTION_MAP.get(loinc_code)
            if target_field:
                text_content = " ".join(section_elem.itertext())
                cleaned = _clean_text(text_content)
                if cleaned:
                    extracted_lists[target_field].append(cleaned)

        sections = {
            k: _clean_text(" ".join(v))
            for k, v in extracted_lists.items()
            if v
        }
        return {
            "sections": sections,
            "label_version_date": published_date,
            "source": "DailyMed",
        }
    except Exception as exc:
        logger.warning(f"DailyMed fallback failed for '{generic_name}': {exc}")
        return {"sections": {}, "label_version_date": "unknown", "source": "DailyMed"}


def fetch_drug_label_chunks(generic_name: str, rxcui: str) -> List[Dict[str, Any]]:
    """
    Pull label sections from openFDA, falling back to DailyMed for any missing key sections.
    Formats each section into the exact chunk metadata format:
    {
      "drug": "ibuprofen",
      "rxcui": "5640",
      "section": "contraindications",
      "text": "...",
      "source": "openFDA / DailyMed",
      "label_version_date": "...",
      "retrieved_on": "YYYY-MM-DD"
    }
    """
    today_str = date.today().isoformat()
    openfda_res = fetch_openfda_sections(generic_name, rxcui)
    sections_map: Dict[str, Dict[str, str]] = {}

    for sec_name, sec_text in openfda_res.get("sections", {}).items():
        sections_map[sec_name] = {
            "text": sec_text,
            "source": "openFDA",
            "label_version_date": openfda_res.get("label_version_date", "unknown"),
        }

    # Check if critical sections are missing; if so, invoke DailyMed SPL XML fallback
    critical_sections = {"dosage_and_administration", "contraindications", "drug_interactions"}
    missing_critical = critical_sections - set(sections_map.keys())

    if missing_critical or len(sections_map) < 3:
        dailymed_res = fetch_dailymed_sections(generic_name)
        for sec_name, sec_text in dailymed_res.get("sections", {}).items():
            if sec_name not in sections_map and sec_text:
                sections_map[sec_name] = {
                    "text": sec_text,
                    "source": "DailyMed (SPL XML)",
                    "label_version_date": dailymed_res.get("label_version_date", "unknown"),
                }

    chunks: List[Dict[str, Any]] = []
    for sec_name in TARGET_LABEL_SECTIONS:
        if sec_name in sections_map:
            info = sections_map[sec_name]
            chunks.append(
                {
                    "drug": generic_name.lower(),
                    "rxcui": str(rxcui),
                    "section": sec_name,
                    "text": info["text"],
                    "source": info["source"],
                    "label_version_date": info["label_version_date"],
                    "retrieved_on": today_str,
                }
            )

    return chunks

