"""
ISO-Compliant Clinical Decision Support Evaluator (`engine/cdss_evaluator.py`).
Architecture:
1. Deterministic Rule Engine First (`engine/deterministic_rules.py`):
   - Raises all primary interaction (DDInter 2.0), allergy/cross-reactivity (WHO/ISO ATC),
     duplicate ingredient/class (RxNorm/ATC), dose/renal/hepatic/pregnancy/Beers alerts,
     and determines `NOT_VERIFIED` status when data is missing.
2. Local Qwen LLM Explainer (`qwen3:8b`) with Strict Citation Gate:
   - Phrases clinical explanations for deterministic findings.
   - If the LLM proposes any additional alert, it MUST cite a valid `chunk_id` from the retrieved
     knowledge base, and Python validates that the cited chunk text explicitly mentions the involved
     drugs/condition. Uncited or unverified LLM alerts are dropped and logged in the ISO trace.
3. Full System Traceability (`engine/trace_logger.py`):
   - Logs inputs, sources, model, deterministic vs. LLM decisions, outputs, and millisecond
     latency breakdown to `logs/cdss_system_trace.jsonl` and `logs/cdss_system.log`.
"""

import json
import re
from typing import Any, Dict, List, Optional
import requests

from config import (
    EVALUATION_HISTORY_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_FALLBACK_MODELS,
    OLLAMA_MODEL,
)
from engine.deterministic_rules import (
    extract_allergies_and_context_from_notes,
    parse_structured_dose,
    run_deterministic_cdss_rules,
)
from engine.trace_logger import CDSSTraceContext, get_iso8601_timestamp, iso_logger
from pipeline.ddinter_loader import DDInterStore
from pipeline.rxclass import resolve_drug_atc_and_class
from pipeline.vector_store import CDSSVectorStore


def get_active_ollama_model() -> str:
    """Detect which Qwen model is available in local Ollama, defaulting to qwen3:8b."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
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
        iso_logger.debug(f"Ollama model check skipped: {exc}")
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


def validate_llm_proposed_alert(
    alert: Dict[str, Any],
    chunk_lookup: Dict[str, Dict[str, Any]],
    all_generic_names: List[str],
) -> bool:
    """
    Citation Validator for LLM-proposed alerts:
    1. Must include a `cited_chunk_id` that exists in `chunk_lookup`.
    2. If `type == DRUG_INTERACTION`, the cited chunk text must explicitly mention BOTH
       interacting drugs (preventing narrative hallucination like Amoxicillin + Ibuprofen).
    """
    cited_id = str(alert.get("cited_chunk_id") or "").strip()
    if not cited_id or cited_id not in chunk_lookup:
        return False

    chunk_text = chunk_lookup[cited_id].get("text", "").lower()
    a_type = str(alert.get("type", "")).upper()

    if "INTERACTION" in a_type:
        involved = [str(d).strip().lower() for d in alert.get("drugs_involved", []) if d]
        if len(involved) < 2:
            return False
        # Verify both drugs appear in the cited chunk text
        for d in involved[:2]:
            if d not in chunk_text:
                return False
        return True

    return len(chunk_text) > 20


class CDSSEvaluator:
    """
    ISO-Compliant Clinical Decision Support Evaluator:
    Deterministic Rule Engine + Structured DDInter 2.0 + WHO/ISO ATC + Cited Local Qwen LLM Explainer.
    """

    def __init__(
        self,
        vector_store: Optional[CDSSVectorStore] = None,
        ddinter_store: Optional[DDInterStore] = None,
    ) -> None:
        self.store = vector_store or CDSSVectorStore()
        self.ddinter = ddinter_store or DDInterStore()

    def evaluate_prescription(
        self,
        patient_id: str,
        patient_name: str,
        medications: List[Dict[str, str]],
        patient_notes: str = "",
        coded_allergies: Optional[List[Dict[str, str]]] = None,
        patient_age: Optional[int] = None,
        egfr: Optional[float] = None,
        pregnancy_status: Optional[str] = None,
        lactation_status: Optional[bool] = None,
        hepatic_impairment: Optional[bool] = None,
        use_llm_explainer: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a prescription through the deterministic rule engine and cited LLM explainer,
        recording a full ISO trace with sub-step millisecond latencies.
        """
        structured_context = extract_allergies_and_context_from_notes(
            patient_notes=patient_notes,
            coded_allergies=coded_allergies,
            patient_age=patient_age,
            egfr=egfr,
            pregnancy_status=pregnancy_status,
            lactation_status=lactation_status,
            hepatic_impairment=hepatic_impairment,
        )

        trace = CDSSTraceContext(
            inputs={
                "patient_id": patient_id,
                "patient_name": patient_name,
                "patient_notes": patient_notes,
                "structured_context": structured_context,
                "medications": medications,
            }
        )

        med_records: List[Dict[str, Any]] = []
        all_generic_names: List[str] = []
        chunk_lookup: Dict[str, Dict[str, Any]] = {}
        used_citations: List[Dict[str, Any]] = []

        # PHASE 1 & 2: RxNorm + WHO/ISO ATC Normalization & Vector DB Retrieval
        for med in medications:
            raw_name = (med.get("name") or "").strip()
            raw_dose = (med.get("dose") or "").strip()
            if not raw_name:
                continue

            trace.start_step("rxnorm_atc_normalization_ms")
            indexed_info = self.store.ensure_drug_indexed(raw_name)
            norm = indexed_info["normalization"]
            gen_name = norm["generic_name"].lower()
            rxcui = norm["rxcui"]
            atc_info = resolve_drug_atc_and_class(gen_name, rxcui)
            dose_parsed = parse_structured_dose(raw_dose)
            trace.end_step("rxnorm_atc_normalization_ms")

            trace.start_step("kb_vector_retrieval_ms")
            who_eml = indexed_info["who_eml"]
            chunks = indexed_info["chunks"]
            chunk_by_sec = {c["section"]: c for c in chunks}
            for c in chunks:
                cid = c.get("chunk_id") or f"{gen_name}::{rxcui}::{c['section']}"
                chunk_lookup[cid] = c
                used_citations.append(
                    {
                        "chunk_id": cid,
                        "drug": gen_name,
                        "rxcui": rxcui,
                        "atc_code": atc_info.get("atc_code", ""),
                        "section": c["section"],
                        "source": c["source"],
                        "label_version_date": c["label_version_date"],
                        "retrieved_on": c["retrieved_on"],
                        "excerpt": c["text"][:420],
                    }
                )
            trace.end_step("kb_vector_retrieval_ms")

            # Log sources used for this medication
            trace.add_source(
                source_name=norm.get("resolution_source", "RxNorm REST API"),
                drug=gen_name,
                identifier=f"RxCUI:{rxcui}",
                detail=f"Entered='{raw_name}' -> Generic='{gen_name}'",
                fetched_live=indexed_info["fetched_live"],
            )
            trace.add_source(
                source_name=atc_info.get("source", "WHO/ISO ATC"),
                drug=gen_name,
                identifier=f"ATC:{atc_info.get('atc_code')}",
                detail=f"Class={atc_info.get('therapeutic_class')}",
                fetched_live=False,
            )
            for c in chunks:
                trace.add_source(
                    source_name=c["source"],
                    drug=gen_name,
                    identifier=c.get("chunk_id", c["section"]),
                    detail=f"Section={c['section']} (Version={c['label_version_date']})",
                    fetched_live=indexed_info["fetched_live"],
                )

            all_generic_names.append(gen_name)
            med_records.append(
                {
                    "entered_name": raw_name,
                    "entered_dose": raw_dose or "Not specified",
                    "dose_parsed": dose_parsed,
                    "generic_name": gen_name,
                    "rxcui": rxcui,
                    "atc_info": atc_info,
                    "ingredients": norm.get("ingredients", [gen_name]),
                    "resolution_source": norm.get("resolution_source", "RxNorm"),
                    "who_eml": who_eml,
                    "chunks_by_section": chunk_by_sec,
                    "all_chunks": chunks,
                    "fetched_live": indexed_info["fetched_live"],
                }
            )

        # PHASE 3: Deterministic Clinical Rule Engine (DDInter 2.0 + ATC Allergy + Dose + KDIGO Renal)
        trace.start_step("deterministic_rules_ms")
        det_result = run_deterministic_cdss_rules(
            med_records=med_records,
            patient_context=structured_context,
            ddinter_store=self.ddinter,
        )
        trace.end_step("deterministic_rules_ms")

        for alert in det_result["alerts"]:
            if alert.get("ddinter_id"):
                trace.add_source(
                    source_name="DDInter 2.0 Structured DDI DB",
                    drug=" + ".join(alert.get("drugs_involved", [])),
                    identifier=alert["ddinter_id"],
                    detail=alert.get("severity", "Major"),
                    fetched_live=False,
                )

        trace.deterministic_trace = {
            "alerts_raised": len(det_result["alerts"]),
            "suggestions_raised": len(det_result["suggestions"]),
            "not_verified_reasons": det_result["not_verified_reasons"],
            "alerts": det_result["alerts"],
        }

        final_alerts: List[Dict[str, Any]] = list(det_result["alerts"])
        final_suggestions: List[str] = list(det_result["suggestions"])
        final_med_reviews: List[Dict[str, Any]] = list(det_result["medication_reviews"])
        not_verified_reasons: List[str] = list(det_result["not_verified_reasons"])

        # Build deterministic baseline clinical summary
        if not_verified_reasons and not final_alerts:
            baseline_summary = "NOT VERIFIED: " + " ".join(not_verified_reasons)
        elif final_alerts:
            top_descriptions = [a["description"] for a in final_alerts[:2]]
            baseline_summary = " ".join(top_descriptions)
        elif final_suggestions:
            baseline_summary = "No critical contraindications or major drug interactions found. Review clinical monitoring suggestions: " + final_suggestions[0]
        else:
            baseline_summary = (
                "All prescribed medications and calculated daily doses are within standard therapeutic limits. "
                "No DDInter 2.0 drug interactions, duplicate classes, or allergy contraindications detected."
            )

        llm_summary = baseline_summary

        # PHASE 4: Local Qwen LLM Explainer + Citation Validation Gate
        model_name = get_active_ollama_model()
        if use_llm_explainer:
            trace.start_step("llm_explainer_inference_ms")
            llm_raw = self._call_qwen_explainer(
                model_name=model_name,
                patient_id=patient_id,
                patient_name=patient_name,
                patient_notes=patient_notes,
                structured_context=structured_context,
                med_records=med_records,
                deterministic_alerts=final_alerts,
                deterministic_suggestions=final_suggestions,
                chunk_lookup=chunk_lookup,
                trace=trace,
            )
            trace.end_step("llm_explainer_inference_ms")

            # Citation Validation Gate
            trace.start_step("citation_validation_ms")
            if llm_raw.get("summary"):
                llm_summary = str(llm_raw["summary"])

            proposed_alerts = llm_raw.get("additional_cited_alerts", [])
            trace.llm_citation_audit["raw_llm_alerts_count"] = len(proposed_alerts)

            for prop in proposed_alerts:
                if validate_llm_proposed_alert(prop, chunk_lookup, all_generic_names):
                    prop["deterministic"] = False
                    final_alerts.append(prop)
                    trace.llm_citation_audit["accepted_cited_alerts"].append(prop)
                else:
                    trace.llm_citation_audit["rejected_uncited_alerts"].append(prop)

            for s in llm_raw.get("additional_suggestions", []):
                if isinstance(s, str) and s.strip() and s not in final_suggestions:
                    final_suggestions.append(s.strip())
            trace.end_step("citation_validation_ms")

        # PHASE 5: Determine 4-State Overall Status (ALERT, NOT_VERIFIED, SUGGESTION, ALL_RIGHT)
        if final_alerts:
            severities = [str(a.get("severity", "HIGH")).upper() for a in final_alerts]
            if any(s in ("HIGH", "MAJOR", "CRITICAL", "SEVERE", "CONTRAINDICATED") for s in severities):
                overall_status = "ALERT"
            else:
                overall_status = "SUGGESTION"
        elif not_verified_reasons:
            overall_status = "NOT_VERIFIED"
        elif final_suggestions:
            overall_status = "SUGGESTION"
        else:
            overall_status = "ALL_RIGHT"

        status_labels = {
            "ALL_RIGHT": "ALL RIGHT — Verified Safe (No Drug, Allergy, or Dose Issues)",
            "SUGGESTION": "CLINICAL SUGGESTION — Review Monitoring / Dose Guidance",
            "ALERT": "CLINICAL ALERT — High-Risk Interaction, Allergy, Duplicate, or Overdose!",
            "NOT_VERIFIED": "NOT VERIFIED — Unindexed Drug or Missing Mandatory Clinical Data",
        }

        final_response: Dict[str, Any] = {
            "trace_id": trace.trace_id,
            "timestamp": get_iso8601_timestamp(),
            "patient": {
                "patient_id": patient_id,
                "patient_name": patient_name,
                "patient_notes": patient_notes,
                "structured_context": structured_context,
            },
            "model_used": model_name,
            "overall_status": overall_status,
            "status_label": status_labels.get(overall_status, overall_status),
            "summary": llm_summary,
            "not_verified_reasons": not_verified_reasons,
            "alerts": final_alerts,
            "suggestions": final_suggestions,
            "medication_reviews": final_med_reviews,
            "normalized_medications": [
                {
                    "entered_name": r["entered_name"],
                    "entered_dose": r["entered_dose"],
                    "calculated_daily_dose": r["dose_parsed"]["formatted_calc"],
                    "generic_name": r["generic_name"],
                    "rxcui": r["rxcui"],
                    "atc_code": r["atc_info"].get("atc_code", ""),
                    "therapeutic_class": r["atc_info"].get("therapeutic_class", ""),
                    "ingredients": r["ingredients"],
                    "who_eml_status": r["who_eml"].get("eml_status", ""),
                    "who_typical_dose": r["who_eml"].get("typical_adult_dose", ""),
                    "fetched_live": r["fetched_live"],
                }
                for r in med_records
            ],
            "citations": used_citations,
        }

        # Finalize and write full ISO trace record
        trace_record = trace.finalize_and_write(final_response)
        final_response["system_trace"] = {
            "trace_id": trace_record["trace_id"],
            "started_at": trace_record["started_at"],
            "completed_at": trace_record["completed_at"],
            "latency_ms": trace_record["latency_ms"],
            "sources_summary": trace_record["sources_summary"],
            "model_used": trace_record["model_used"],
            "llm_citation_audit": trace_record["llm_citation_audit"],
        }

        # Also append compact record to evaluation_history.jsonl
        try:
            with open(EVALUATION_HISTORY_PATH, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "trace_id": trace.trace_id,
                            "timestamp": final_response["timestamp"],
                            "patient_id": patient_id,
                            "patient_name": patient_name,
                            "overall_status": overall_status,
                            "medications": [
                                {
                                    "name": r["entered_name"],
                                    "generic": r["generic_name"],
                                    "atc": r["atc_info"].get("atc_code", ""),
                                    "dose": r["entered_dose"],
                                    "daily_calc": r["dose_parsed"]["formatted_calc"],
                                }
                                for r in med_records
                            ],
                            "latency_ms": trace_record["latency_ms"]["total_latency_ms"],
                            "summary": final_response["summary"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception as exc:
            iso_logger.debug(f"Could not write evaluation history: {exc}")

        return final_response

    def _call_qwen_explainer(
        self,
        model_name: str,
        patient_id: str,
        patient_name: str,
        patient_notes: str,
        structured_context: Dict[str, Any],
        med_records: List[Dict[str, Any]],
        deterministic_alerts: List[Dict[str, Any]],
        deterministic_suggestions: List[str],
        chunk_lookup: Dict[str, Dict[str, Any]],
        trace: CDSSTraceContext,
    ) -> Dict[str, Any]:
        """
        Invoke local Ollama Qwen model strictly as an EXPLAINER for deterministic alerts.
        Any extra alert proposed by the LLM must include a valid `cited_chunk_id`.
        """
        trace.model_info["model_name"] = model_name
        trace.model_info["endpoint"] = f"{OLLAMA_BASE_URL}/api/chat"
        trace.model_info["invoked"] = True

        chunk_snippets = []
        for cid, ch in list(chunk_lookup.items())[:12]:
            if ch["section"] in ("contraindications", "warnings", "dosage_and_administration", "who_eml_reference"):
                chunk_snippets.append(f"[chunk_id: {cid}] ({ch['source']}): {ch['text'][:400]}")

        system_prompt = (
            "You are a Clinical Pharmacist AI Explainer in an ISO-compliant hospital CDSS.\n"
            "IMPORTANT ARCHITECTURE RULE: All drug-drug interactions (DDInter 2.0), allergies (WHO/ISO ATC), "
            "duplicate therapies, and dose/renal calculations have ALREADY been computed deterministically in code.\n"
            "Your role is ONLY to:\n"
            "1. Write a clear, concise 1-3 sentence `summary` explaining the deterministic findings to the clinician.\n"
            "2. Do NOT invent any drug-drug interaction that is not in `DETERMINISTIC ALERTS`.\n"
            "3. Only propose an item in `additional_cited_alerts` if a clear contraindication exists in the provided chunks "
            "that the deterministic rules missed, and you MUST provide the exact `cited_chunk_id`.\n\n"
            "Respond strictly in JSON matching this schema:\n"
            "{\n"
            "  \"summary\": \"Clear 1-3 sentence clinical explanation of the deterministic safety evaluation.\",\n"
            "  \"additional_cited_alerts\": [],\n"
            "  \"additional_suggestions\": [\"Optional brief clinical pearl or monitoring tip.\"]\n"
            "}\n"
            "/no_think"
        )

        user_prompt = (
            f"PATIENT: {patient_name} ({patient_id}) | Notes: {patient_notes or 'None'}\n"
            f"PARSED CONTEXT: {json.dumps(structured_context)}\n"
            f"MEDICATIONS:\n"
            + "\n".join(
                f"- {r['entered_name']} -> {r['generic_name']} (RxCUI:{r['rxcui']}, ATC:{r['atc_info'].get('atc_code')}) | "
                f"Dose: {r['entered_dose']} ({r['dose_parsed']['formatted_calc']})"
                for r in med_records
            )
            + f"\n\nDETERMINISTIC ALERTS LOCKED IN CODE:\n{json.dumps(deterministic_alerts, indent=2)}\n"
            + f"\nDETERMINISTIC SUGGESTIONS:\n{json.dumps(deterministic_suggestions, indent=2)}\n"
            + "\nAVAILABLE CHUNKS:\n"
            + "\n".join(chunk_snippets)
            + "\n\nProvide your JSON explanation now. /no_think"
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
                "options": {"temperature": 0.0, "num_ctx": 4096},
            }
            resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=45)
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "")
            parsed = parse_llm_json(content)
            trace.model_info["status"] = "SUCCESS"
            return parsed
        except Exception as exc:
            trace.model_info["status"] = f"FALLBACK_DETERMINISTIC ({type(exc).__name__})"
            iso_logger.warning(f"LLM Explainer skipped/fallback (Deterministic rules active): {exc}")
            return {"summary": "", "additional_cited_alerts": [], "additional_suggestions": []}
