"""
ISO-Compliant System Trace & Audit Logger (ISO 8601 / ISO 14971 Traceability).
Records every CDSS transaction to both:
1. `logs/cdss_system_trace.jsonl` (Structured JSON Lines with full inputs, sources, model, output, and ms latency breakdown)
2. `logs/cdss_system.log` (Human-readable ISO-8601 formatted operational log)
"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import SYSTEM_LOG_FILE_PATH, SYSTEM_TRACE_JSONL_PATH


def get_iso8601_timestamp() -> str:
    """Return current local timestamp in strict ISO 8601 format with millisecond precision and timezone offset."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def setup_iso_file_logger() -> logging.Logger:
    """Configure application-wide file and console logging with ISO 8601 timestamps."""
    logger = logging.getLogger("CDSS_ISO_TRACE")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    file_handler = logging.FileHandler(str(SYSTEM_LOG_FILE_PATH), encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


iso_logger = setup_iso_file_logger()


class CDSSTraceContext:
    """
    Context tracker for a single CDSS evaluation request.
    Measures sub-step latencies in milliseconds, tracks all external/local sources used,
    records deterministic vs. LLM decisions, and writes an immutable trace entry.
    """

    def __init__(self, inputs: Dict[str, Any]) -> None:
        self.trace_id: str = f"CDSS-{uuid.uuid4()}"
        self.started_at_iso: str = get_iso8601_timestamp()
        self._start_perf: float = time.perf_counter()
        self._step_timers: Dict[str, float] = {}

        self.inputs: Dict[str, Any] = inputs
        self.latency_ms: Dict[str, float] = {
            "rxnorm_atc_normalization_ms": 0.0,
            "kb_vector_retrieval_ms": 0.0,
            "deterministic_rules_ms": 0.0,
            "llm_explainer_inference_ms": 0.0,
            "citation_validation_ms": 0.0,
            "total_latency_ms": 0.0,
        }
        self.sources_used: List[Dict[str, Any]] = []
        self.model_info: Dict[str, Any] = {
            "model_name": "none",
            "endpoint": "none",
            "role": "EXPLAINER_WITH_CITATION_GATE",
            "invoked": False,
            "status": "NOT_CALLED",
        }
        self.deterministic_trace: Dict[str, Any] = {}
        self.llm_citation_audit: Dict[str, Any] = {
            "raw_llm_alerts_count": 0,
            "accepted_cited_alerts": [],
            "rejected_uncited_alerts": [],
        }

    def start_step(self, step_name: str) -> None:
        """Start high-resolution timer for a named pipeline phase."""
        self._step_timers[step_name] = time.perf_counter()

    def end_step(self, step_name: str) -> float:
        """Stop timer for a named pipeline phase and store elapsed milliseconds."""
        start_t = self._step_timers.get(step_name)
        if start_t is None:
            return 0.0
        elapsed_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
        self.latency_ms[step_name] = round(self.latency_ms.get(step_name, 0.0) + elapsed_ms, 2)
        return elapsed_ms

    def add_source(
        self,
        source_name: str,
        drug: str,
        identifier: str,
        detail: str = "",
        fetched_live: bool = False,
    ) -> None:
        """Record a knowledge source consulted during evaluation."""
        self.sources_used.append(
            {
                "source": source_name,
                "drug": drug,
                "identifier": identifier,
                "detail": detail,
                "mode": "LIVE_API" if fetched_live else "LOCAL_DB",
            }
        )

    def finalize_and_write(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Compute total latency, write structured trace to JSONL and log file, and return trace summary."""
        total_ms = round((time.perf_counter() - self._start_perf) * 1000.0, 2)
        self.latency_ms["total_latency_ms"] = total_ms
        completed_at_iso = get_iso8601_timestamp()

        unique_source_names = sorted({s["source"] for s in self.sources_used})

        trace_record: Dict[str, Any] = {
            "trace_id": self.trace_id,
            "standards": ["ISO-8601:2019", "ISO/IEC-25010", "WHO-ATC", "NLM-RxNorm", "DDInter-2.0"],
            "started_at": self.started_at_iso,
            "completed_at": completed_at_iso,
            "inputs": self.inputs,
            "model_used": self.model_info,
            "sources_summary": unique_source_names,
            "sources_detailed": self.sources_used,
            "deterministic_trace": self.deterministic_trace,
            "llm_citation_audit": self.llm_citation_audit,
            "output_summary": {
                "overall_status": output.get("overall_status"),
                "alerts_count": len(output.get("alerts", [])),
                "suggestions_count": len(output.get("suggestions", [])),
                "not_verified_reasons": output.get("not_verified_reasons", []),
                "alerts": output.get("alerts", []),
                "suggestions": output.get("suggestions", []),
            },
            "latency_ms": self.latency_ms,
        }

        try:
            with open(SYSTEM_TRACE_JSONL_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_record, ensure_ascii=False) + "\n")
        except Exception as exc:
            iso_logger.error(f"Failed writing JSONL trace {self.trace_id}: {exc}")

        med_names = [m.get("name", "") for m in self.inputs.get("medications", [])]
        iso_logger.info(
            f"TRACE={self.trace_id} | PATIENT={self.inputs.get('patient_id')} | "
            f"MEDS={med_names} | STATUS={output.get('overall_status')} | "
            f"ALERTS={len(output.get('alerts', []))} | MODEL={self.model_info.get('model_name')} "
            f"({self.model_info.get('status')}) | SOURCES={unique_source_names} | "
            f"LATENCY_TOTAL={total_ms}ms (RxNorm/ATC={self.latency_ms.get('rxnorm_atc_normalization_ms')}ms, "
            f"KB={self.latency_ms.get('kb_vector_retrieval_ms')}ms, "
            f"Rules={self.latency_ms.get('deterministic_rules_ms')}ms, "
            f"LLM={self.latency_ms.get('llm_explainer_inference_ms')}ms)"
        )

        return trace_record


def read_recent_traces(limit: int = 25) -> List[Dict[str, Any]]:
    """Read the most recent system trace records from `logs/cdss_system_trace.jsonl`."""
    if not SYSTEM_TRACE_JSONL_PATH.exists():
        return []
    try:
        with open(SYSTEM_TRACE_JSONL_PATH, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        recent = lines[-limit:]
        recent.reverse()
        return [json.loads(line) for line in recent]
    except Exception as exc:
        iso_logger.warning(f"Could not read system traces: {exc}")
        return []

