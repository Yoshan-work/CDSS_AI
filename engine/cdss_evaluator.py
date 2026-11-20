import json
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import requests

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_FALLBACK_MODELS,
    EVALUATION_HISTORY_PATH,
)
from pipeline.vector_store import CDSSVectorStore

logger = logging.getLogger(__name__)

# Pharmacologic class keywords to improve cross-referencing inside narrative FDA drug_interactions text
DRUG_CLASS_KEYWORDS: Dict[str, List[str]] = {
    "ibuprofen": ["nsaid", "nonsteroidal anti-inflammatory", "aspirin", "anticoagulant", "warfarin", "diuretic", "ace inhibitor"],
    "diclofenac": ["nsaid", "nonsteroidal anti-inflammatory", "aspirin", "anticoagulant", "warfarin", "diuretic", "ace inhibitor"],
    "naproxen": ["nsaid", "nonsteroidal anti-inflammatory", "aspirin", "anticoagulant", "warfarin", "diuretic", "ace inhibitor"],
    "celecoxib": ["nsaid", "nonsteroidal anti-inflammatory", "anticoagulant", "warfarin", "fluconazole"],
    "aspirin": ["nsaid", "salicylate", "antiplatelet", "anticoagulant", "warfarin", "ibuprofen", "clopidogrel", "corticosteroid"],
    "warfarin": ["anticoagulant", "nsaid", "aspirin", "antiplatelet", "metronidazole", "fluconazole", "ciprofloxacin", "amiodarone", "macrolide"],
    "clopidogrel": ["antiplatelet", "omeprazole", "esomeprazole", "cyp2c19", "nsaid", "anticoagulant", "warfarin", "aspirin"],
    "apixaban": ["anticoagulant", "nsaid", "aspirin", "antiplatelet", "clopidogrel"],
    "rivaroxaban": ["anticoagulant", "nsaid", "aspirin", "antiplatelet", "clopidogrel"],
    "omeprazole": ["proton pump inhibitor", "cyp2c19", "clopidogrel", "methotrexate", "digoxin"],
    "esomeprazole": ["proton pump inhibitor", "cyp2c19", "clopidogrel"],
    "enalapril": ["ace inhibitor", "angiotensin", "spironolactone", "potassium", "nsaid", "diuretic", "losartan", "lithium"],
    "lisinopril": ["ace inhibitor", "angiotensin", "spironolactone", "potassium", "nsaid", "diuretic", "losartan", "lithium"],
    "losartan": ["angiotensin receptor blocker", "arb", "spironolactone", "potassium", "nsaid", "enalapril", "lisinopril", "lithium"],
    "spironolactone": ["potassium-sparing", "hyperkalemia", "ace inhibitor", "enalapril", "lisinopril", "losartan", "nsaid"],
    "simvastatin": ["statin", "hmg-coa", "cyp3a4", "amlodipine", "amiodarone", "clarithromycin", "itraconazole", "gemfibrozil"],
    "atorvastatin": ["statin", "hmg-coa", "cyp3a4", "clarithromycin", "itraconazole", "cyclosporine"],
    "amlodipine": ["calcium channel blocker", "simvastatin", "cyp3a4"],
    "metformin": ["biguanide", "lactic acidosis", "contrast", "renal", "cimetidine", "furosemide"],
    "ciprofloxacin": ["fluoroquinolone", "warfarin", "theophylline", "tizanidine", "qt prolongation", "corticosteroid"],
    "metronidazole": ["warfarin", "alcohol", "disulfiram", "lithium", "busulfan"],
    "clarithromycin": ["macrolide", "cyp3a4", "statin", "simvastatin", "atorvastatin", "warfarin", "digoxin"],
    "azithromycin": ["macrolide", "qt prolongation", "amiodarone", "warfarin", "digoxin"],
    "prednisolone": ["corticosteroid", "nsaid", "aspirin", "warfarin", "fluoroquinolone", "diabetes"],
}


def extract_focused_snippet(text: str, keywords: List[str], window: int = 450, max_len: int = 1200) -> str:
    """
    Extract paragraphs/windows from a long FDA label section that match any of the target
    keywords (e.g., interacting drug names or pharmacologic classes), falling back to the start of the text.
    """
    if not text:
        return ""
    lower_text = text.lower()
    snippets: List[str] = []

    for kw in keywords:
        if not kw or len(kw) < 3:
            continue
        idx = lower_text.find(kw.lower())
        if idx != -1:
            start = max(0, idx - 120)
            end = min(len(text), idx + window)
            snippet = text[start:end].strip()
            if snippet not in snippets:
                snippets.append(f"...{snippet}...")
        if sum(len(s) for s in snippets) >= max_len:
            break

    if snippets:
        return " ".join(snippets)[:max_len]
    return text[:max_len]


def get_active_ollama_model() -> str:
    """Detect which Qwen model is available in local Ollama, defaulting to qwen3:8b."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.ok:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            if OLLAMA_MODEL in models:
                return OLLAMA_MODEL
            for candidate in OLLAMA_FALLBACK_MODELS:
                if candidate in models:
                    return candidate
            for m in models:
                if "qwen" in m.lower():
                    return m
            if models:
                return models[0]
    except Exception as exc:
        logger.debug(f"Could not list Ollama models: {exc}")
    return OLLAMA_MODEL


def parse_llm_json(raw_content: str) -> Dict[str, Any]:
    """Strip any <think>...</think> blocks or markdown fences and parse JSON safely."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Could not parse valid JSON from LLM output: {cleaned[:300]}")


class CDSSEvaluator:
    """
    Clinical Decision Support Evaluator combining:
    1. RxNorm brand/generic/RxCUI normalization & duplicate ingredient detection
    2. openFDA + DailyMed + WHO EML RAG chunk retrieval
    3. Local Qwen LLM (Ollama) clinical evaluation with structured Alert / Suggestion / All Right output
    """

    def __init__(self, vector_store: Optional[CDSSVectorStore] = None):
        self.store = vector_store or CDSSVectorStore()

    def evaluate_prescription(
        self,
        patient_id: str,
        patient_name: str,
        medications: List[Dict[str, str]],
        patient_notes: str = "",
    ) -> Dict[str, Any]:
        """
        Evaluate a patient's medication list and doses.
        medications: [{"name": "Brufen", "dose": "400mg TID"}, ...]
        """
        med_records: List[Dict[str, Any]] = []
        all_generic_names: List[str] = []

        # Step 1: Normalize all drugs via RxNorm & ensure chunks are in Vector DB
        for med in medications:
            raw_name = (med.get("name") or "").strip()
            raw_dose = (med.get("dose") or "").strip()
            if not raw_name:
                continue

            indexed_info = self.store.ensure_drug_indexed(raw_name)
            norm = indexed_info["normalization"]
            who_eml = indexed_info["who_eml"]
            chunks = indexed_info["chunks"]

            gen_name = norm["generic_name"]
            all_generic_names.append(gen_name)

            chunk_by_sec = {c["section"]: c for c in chunks}
            med_records.append(
                {
                    "entered_name": raw_name,
                    "entered_dose": raw_dose or "Not specified",
                    "generic_name": gen_name,
                    "rxcui": norm["rxcui"],
                    "ingredients": norm.get("ingredients", [gen_name]),
                    "resolution_source": norm.get("resolution_source", "RxNorm"),
                    "who_eml": who_eml,
                    "chunks_by_section": chunk_by_sec,
                    "all_chunks": chunks,
                    "fetched_live": indexed_info["fetched_live"],
                }
            )

        # Step 2: Deterministic Pre-Check for Duplicate Active Ingredients (e.g., Brufen + Advil -> both Ibuprofen)
        duplicate_warnings: List[Dict[str, Any]] = []
        seen_rxcuis: Dict[str, List[str]] = {}
        for rec in med_records:
            key = rec["rxcui"] if rec["rxcui"] != "UNKNOWN" else rec["generic_name"]
            seen_rxcuis.setdefault(key, []).append(f"{rec['entered_name']} ({rec['entered_dose']})")

        for key, entered_items in seen_rxcuis.items():
            if len(entered_items) > 1:
                gen = next(
                    (r["generic_name"] for r in med_records if (r["rxcui"] == key or r["generic_name"] == key)),
                    key,
                )
                duplicate_warnings.append(
                    {
                        "severity": "HIGH",
                        "type": "DUPLICATE_THERAPY",
                        "drugs_involved": entered_items,
                        "description": (
                            f"Duplicate active ingredient detected via RxNorm (RxCUI {key}): "
                            f"{' and '.join(entered_items)} both contain '{gen}'. "
                            f"Co-administration risks accidental overdose and toxicity."
                        ),
                        "source": f"RxNorm Normalization (RxCUI: {key})",
                    }
                )

        # Step 3: Build Focused RAG Context for Local Qwen LLM
        rag_evidence_blocks: List[str] = []
        used_citations: List[Dict[str, Any]] = []

        for idx, rec in enumerate(med_records, start=1):
            gen = rec["generic_name"]
            rxcui = rec["rxcui"]
            other_drugs = [g for g in all_generic_names if g != gen]
            search_keywords = list(other_drugs) + DRUG_CLASS_KEYWORDS.get(gen, [])
            for od in other_drugs:
                search_keywords.extend(DRUG_CLASS_KEYWORDS.get(od, []))

            sec_map = rec["chunks_by_section"]
            who = rec["who_eml"]

            drug_block_lines = [
                f"=== MEDICATION #{idx}: Entered='{rec['entered_name']}' | Prescribed Dose='{rec['entered_dose']}' ===",
                f"Normalized Generic Name: {gen} | RxNorm RxCUI: {rxcui} | Active Ingredients: {', '.join(rec['ingredients'])}",
                f"WHO EML Status: {who.get('eml_status', 'N/A')} | WHO Reference Adult Dose: {who.get('typical_adult_dose') or 'See FDA Label'}",
            ]
            if who.get("who_notes"):
                drug_block_lines.append(f"WHO EML Clinical Notes: {who.get('who_notes')}")

            # Include focused snippets from openFDA / DailyMed chunks
            for sec_name in [
                "dosage_and_administration",
                "drug_interactions",
                "contraindications",
                "warnings",
                "pregnancy",
                "indications_and_usage",
            ]:
                ch = sec_map.get(sec_name)
                if not ch:
                    continue
                if sec_name == "drug_interactions":
                    snippet = extract_focused_snippet(ch["text"], search_keywords, window=500, max_len=1100)
                elif sec_name == "dosage_and_administration":
                    snippet = extract_focused_snippet(
                        ch["text"], ["adult", "usual", "recommended", "maximum", "mg", "daily"], window=500, max_len=900
                    )
                else:
                    snippet = extract_focused_snippet(
                        ch["text"], search_keywords + [patient_notes], window=400, max_len=650
                    )

                drug_block_lines.append(
                    f"[{sec_name.upper()} | Source: {ch['source']} | Version: {ch['label_version_date']}]: {snippet}"
                )
                used_citations.append(
                    {
                        "drug": gen,
                        "rxcui": rxcui,
                        "section": sec_name,
                        "source": ch["source"],
                        "label_version_date": ch["label_version_date"],
                        "retrieved_on": ch["retrieved_on"],
                        "excerpt": snippet[:500],
                    }
                )

            rag_evidence_blocks.append("\n".join(drug_block_lines))

        # Step 4: Call Local Qwen LLM via Ollama
        model_name = get_active_ollama_model()
        llm_result = self._call_qwen_llm(
            model_name=model_name,
            patient_id=patient_id,
            patient_name=patient_name,
            patient_notes=patient_notes,
            med_records=med_records,
            duplicate_warnings=duplicate_warnings,
            rag_context="\n\n".join(rag_evidence_blocks),
        )

        # Merge deterministic duplicate warnings if LLM missed any
        existing_types = {a.get("type", "") for a in llm_result.get("alerts", [])}
        if duplicate_warnings and "DUPLICATE_THERAPY" not in existing_types:
            llm_result.setdefault("alerts", []).insert(0, duplicate_warnings[0])
            llm_result["overall_status"] = "ALERT"

        # Normalize overall_status to strictly one of: ALL_RIGHT, SUGGESTION, ALERT
        raw_status = str(llm_result.get("overall_status", "ALL_RIGHT")).upper().replace(" ", "_")
        if "ALERT" in raw_status or "DANGER" in raw_status or "HIGH" in raw_status:
            overall_status = "ALERT"
        elif "SUGGEST" in raw_status or "WARN" in raw_status or "MODERATE" in raw_status or "CAUTION" in raw_status:
            overall_status = "SUGGESTION"
        else:
            overall_status = "ALL_RIGHT"

        # Ensure if there are high/critical alerts, status is ALERT; if minor alerts/suggestions only, check appropriately
        if llm_result.get("alerts"):
            severities = [str(a.get("severity", "")).upper() for a in llm_result["alerts"]]
            if any(s in ("HIGH", "CRITICAL", "SEVERE", "CONTRAINDICATED") for s in severities):
                overall_status = "ALERT"
            elif overall_status == "ALL_RIGHT":
                overall_status = "SUGGESTION"

        final_response = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "patient": {
                "patient_id": patient_id,
                "patient_name": patient_name,
                "patient_notes": patient_notes,
            },
            "model_used": model_name,
            "overall_status": overall_status,
            "status_label": (
                "ALL RIGHT — No Critical Drug/Dose Issues Detected"
                if overall_status == "ALL_RIGHT"
                else (
                    "CLINICAL SUGGESTION — Review Dose / Monitoring Advice"
                    if overall_status == "SUGGESTION"
                    else "CLINICAL ALERT — High-Risk Interaction, Duplicate, or Dose Issue!"
                )
            ),
            "summary": llm_result.get("summary", ""),
            "alerts": llm_result.get("alerts", []),
            "suggestions": llm_result.get("suggestions", []),
            "medication_reviews": llm_result.get("medication_reviews", []),
            "normalized_medications": [
                {
                    "entered_name": r["entered_name"],
                    "entered_dose": r["entered_dose"],
                    "generic_name": r["generic_name"],
                    "rxcui": r["rxcui"],
                    "ingredients": r["ingredients"],
                    "who_eml_status": r["who_eml"].get("eml_status", ""),
                    "who_typical_dose": r["who_eml"].get("typical_adult_dose", ""),
                    "fetched_live": r["fetched_live"],
                }
                for r in med_records
            ],
            "citations": used_citations,
        }

        # Log evaluation to history JSONL
        try:
            with open(EVALUATION_HISTORY_PATH, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "timestamp": final_response["timestamp"],
                            "patient_id": patient_id,
                            "patient_name": patient_name,
                            "overall_status": overall_status,
                            "medications": [
                                {"name": r["entered_name"], "generic": r["generic_name"], "dose": r["entered_dose"]}
                                for r in med_records
                            ],
                            "summary": final_response["summary"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception as exc:
            logger.debug(f"Could not write evaluation history: {exc}")

        return final_response

    def _call_qwen_llm(
        self,
        model_name: str,
        patient_id: str,
        patient_name: str,
        patient_notes: str,
        med_records: List[Dict[str, Any]],
        duplicate_warnings: List[Dict[str, Any]],
        rag_context: str,
    ) -> Dict[str, Any]:
        """Invoke local Ollama Qwen model with structured clinical JSON output."""
        system_prompt = (
            "You are an expert Hospital Clinical Decision Support System (CDSS) and Senior Clinical Pharmacist AI.\n"
            "Your task is to evaluate a patient's prescribed medications and doses using ONLY the provided "
            "RxNorm normalization data, openFDA / DailyMed official drug label sections, and WHO Essential Medicines List (EML) reference.\n\n"
            "STRICT EVALUATION RULES:\n"
            "1. CHECK DUPLICATE THERAPY: Check if two prescribed drugs share the same normalized generic ingredient / RxCUI (e.g., Brufen and Advil are both ibuprofen).\n"
            "2. CHECK ALLERGIES & CROSS-REACTIVITY: If Patient Notes mention any drug allergy (e.g., Penicillin allergy, Sulfa allergy, NSAID/Aspirin allergy, or a specific drug name), check if any prescribed drug or its pharmacologic class (e.g., Amoxicillin is a penicillin; Brufen/Diclofenac are NSAIDs) triggers that allergy. Flag as HIGH severity CONTRAINDICATION / ALLERGY ALERT.\n"
            "3. CHECK DOSING: Compare each drug's prescribed dose and frequency against the WHO EML Reference Adult Dose and openFDA/DailyMed DOSAGE_AND_ADMINISTRATION text. Flag overdoses, supratherapeutic doses, or unusual regimens.\n"
            "4. CHECK DRUG-DRUG INTERACTIONS: Check if any pair of prescribed medications interacts (e.g., Warfarin + NSAID/Ibuprofen/Aspirin/Metronidazole/Ciprofloxacin; Clopidogrel + Omeprazole; ACEi/ARB + Spironolactone + NSAID; Statin/Simvastatin + Amlodipine/Clarithromycin).\n"
            "5. CHECK CONTRAINDICATIONS & WARNINGS: If patient notes/conditions are provided (e.g. pregnancy, renal impairment/CKD, peptic ulcer, asthma, liver disease), check against CONTRAINDICATIONS, WARNINGS, and PREGNANCY sections.\n"
            "6. DETERMINE OVERALL STATUS:\n"
            "   - \"ALL_RIGHT\": All doses are within standard therapeutic range, no duplicate ingredients, no significant drug-drug interactions, and no allergies/contraindications.\n"
            "   - \"SUGGESTION\": Minor/moderate interaction requiring monitoring, suboptimal dosing, or general administration advice (e.g. take with food, monitor renal function/electrolytes).\n"
            "   - \"ALERT\": Serious/major drug-drug interaction, patient allergy/cross-allergy match, duplicate active ingredient, overdose/exceeding max daily dose, or contraindication.\n\n"
            "Return ONLY a valid JSON object matching this exact schema:\n"
            "{\n"
            "  \"overall_status\": \"ALL_RIGHT\" | \"SUGGESTION\" | \"ALERT\",\n"
            "  \"summary\": \"Clear 1-3 sentence clinical summary of the prescription safety check.\",\n"
            "  \"alerts\": [\n"
            "    {\n"
            "      \"severity\": \"HIGH\" | \"MODERATE\",\n"
            "      \"type\": \"DRUG_INTERACTION\" | \"ALLERGY\" | \"OVERDOSE\" | \"DUPLICATE_THERAPY\" | \"CONTRAINDICATION\",\n"
            "      \"drugs_involved\": [\"drug1\", \"drug2\"],\n"
            "      \"description\": \"Detailed explanation of the clinical risk based on the label text.\",\n"
            "      \"source\": \"openFDA / DailyMed / WHO EML / RxNorm\"\n"
            "    }\n"
            "  ],\n"
            "  \"suggestions\": [\n"
            "    \"Specific actionable clinical suggestion or safer alternative for the prescriber/pharmacist.\"\n"
            "  ],\n"
            "  \"medication_reviews\": [\n"
            "    {\n"
            "      \"entered_name\": \"...\",\n"
            "      \"generic_name\": \"...\",\n"
            "      \"rxcui\": \"...\",\n"
            "      \"entered_dose\": \"...\",\n"
            "      \"dose_status\": \"OK\" | \"HIGH\" | \"LOW\" | \"REVIEW\",\n"
            "      \"comment\": \"Concise assessment of this drug's dose and safety.\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "/no_think"
        )

        prescribed_summary = "\n".join(
            f"- {r['entered_name']} (Normalized: {r['generic_name']}, RxCUI: {r['rxcui']}) — Prescribed Dose: {r['entered_dose']}"
            for r in med_records
        )
        dup_text = (
            json.dumps(duplicate_warnings, indent=2)
            if duplicate_warnings
            else "None detected."
        )

        user_prompt = (
            f"PATIENT INFORMATION:\n"
            f"- Patient ID: {patient_id}\n"
            f"- Patient Name: {patient_name}\n"
            f"- Clinical Notes / Conditions: {patient_notes or 'None reported'}\n\n"
            f"PRESCRIBED MEDICATIONS:\n{prescribed_summary}\n\n"
            f"AUTOMATED RXNORM DUPLICATE CHECK:\n{dup_text}\n\n"
            f"RETRIEVED KNOWLEDGE BASE EVIDENCE (openFDA, DailyMed, WHO EML):\n"
            f"{rag_context}\n\n"
            f"Evaluate this prescription now and respond strictly in JSON. /no_think"
        )

        try:
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "think": False,
                "format": "json",
                "options": {
                    "temperature": 0.1,
                    "num_ctx": 8192,
                },
            }
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=90,
            )
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "")
            return parse_llm_json(content)
        except Exception as exc:
            logger.error(f"Local Qwen LLM invocation error: {exc}")
            # Graceful fallback if Ollama is temporarily unreachable
            return {
                "overall_status": "ALERT" if duplicate_warnings else "SUGGESTION",
                "summary": (
                    f"RxNorm & openFDA knowledge retrieved, but local Ollama model ({model_name}) "
                    f"returned an error: {exc}. Showing deterministic checks."
                ),
                "alerts": duplicate_warnings,
                "suggestions": [
                    "Verify local Ollama server is running (`ollama serve`) with model `qwen3:8b`."
                ],
                "medication_reviews": [
                    {
                        "entered_name": r["entered_name"],
                        "generic_name": r["generic_name"],
                        "rxcui": r["rxcui"],
                        "entered_dose": r["entered_dose"],
                        "dose_status": "REVIEW",
                        "comment": f"Normalized to {r['generic_name']} (RxCUI {r['rxcui']}). Standard dose: {r['who_eml'].get('typical_adult_dose', 'See label')}",
                    }
                    for r in med_records
                ],
            }

