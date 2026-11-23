"""
ISO-Compliant FastAPI Backend for Clinical Decision Support System (`app.py`).
Provides REST endpoints for:
- `POST /api/check` (Deterministic + Cited LLM evaluation with full ISO trace & latency breakdown)
- `GET /api/logs` (Inspect recent system trace entries from `logs/cdss_system_trace.jsonl`)
- `GET /api/status` (System health, active LLM model, Vector DB & DDInter 2.0 statistics)
- `GET /api/drug/{drug_name}` (Inspect RxNorm, ATC, WHO EML, and openFDA/DailyMed chunks)
- `POST /api/pharmacist/verify` (Pharmacist chunk spot-check sign-off)
- `POST /api/seed` (Pre-load hospital formulary drugs)
"""

from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine.cdss_evaluator import CDSSEvaluator, get_active_ollama_model
from engine.trace_logger import iso_logger, read_recent_traces
from pipeline.ddinter_loader import DDInterStore
from pipeline.seed_formulary import PRIORITY_STARTER_DRUGS, seed_formulary_drugs
from pipeline.vector_store import CDSSVectorStore

app = FastAPI(
    title="CDSS — ISO-Compliant Clinical Decision Support System (Deterministic Rules + DDInter 2.0 + WHO/ISO ATC + Cited Qwen LLM)",
    version="2.0.0",
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

vector_store = CDSSVectorStore()
ddinter_store = DDInterStore()
evaluator = CDSSEvaluator(vector_store=vector_store, ddinter_store=ddinter_store)


class MedicationItem(BaseModel):
    name: str = Field(..., description="Brand or generic medication name (e.g., Brufen, Warfarin, Panadol)")
    dose: str = Field(..., description="Dose and frequency (e.g., 400mg TID, 5mg OD)")


class CodedAllergyItem(BaseModel):
    allergen: str = Field(..., description="Coded allergen key (e.g., penicillin, sulfonamide, nsaid, cephalosporin, aspirin)")
    severity: str = Field(default="anaphylaxis_severe", description="anaphylaxis_severe | urticaria_moderate | gi_upset_intolerance")


class PrescriptionCheckRequest(BaseModel):
    patient_id: str = Field(..., description="Patient Identifier")
    patient_name: str = Field(..., description="Patient Full Name")
    patient_notes: Optional[str] = Field(default="", description="Optional clinical notes")
    coded_allergies: Optional[List[CodedAllergyItem]] = Field(default=None, description="Structured coded allergies")
    patient_age: Optional[int] = Field(default=None, description="Patient age in years")
    egfr: Optional[float] = Field(default=None, description="Estimated GFR (mL/min/1.73m2)")
    pregnancy_status: Optional[str] = Field(default="none", description="none | pregnant | trimester_2 | trimester_3")
    lactation_status: Optional[bool] = Field(default=False, description="Whether patient is breastfeeding")
    hepatic_impairment: Optional[bool] = Field(default=False, description="Known active liver disease / cirrhosis")
    medications: List[MedicationItem]


class PharmacistVerifyRequest(BaseModel):
    chunk_id: str
    verified: bool = True
    notes: str = ""


class SeedRequest(BaseModel):
    drugs: Optional[List[str]] = None
    count: int = 10


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status")
def system_status() -> Dict:
    stats = vector_store.get_stats()
    stats["ddinter_records"] = ddinter_store.get_total_records()
    active_model = get_active_ollama_model()
    return {
        "status": "ok",
        "standards": ["ISO-8601:2019", "ISO/IEC-25010", "WHO/ISO-ATC", "DDInter-2.0", "NLM-RxNorm"],
        "ollama_model": active_model,
        "knowledge_base": stats,
    }


@app.post("/api/check")
def check_prescription(req: PrescriptionCheckRequest) -> Dict:
    valid_meds = [
        {"name": m.name.strip(), "dose": m.dose.strip()}
        for m in req.medications
        if m.name.strip()
    ]
    if not valid_meds:
        raise HTTPException(status_code=400, detail="Please enter at least one medication.")

    coded_alg = (
        [{"allergen": a.allergen, "severity": a.severity} for a in req.coded_allergies]
        if req.coded_allergies
        else None
    )

    return evaluator.evaluate_prescription(
        patient_id=req.patient_id.strip() or "N/A",
        patient_name=req.patient_name.strip() or "Unknown Patient",
        medications=valid_meds,
        patient_notes=(req.patient_notes or "").strip(),
        coded_allergies=coded_alg,
        patient_age=req.patient_age,
        egfr=req.egfr,
        pregnancy_status=req.pregnancy_status,
        lactation_status=req.lactation_status,
        hepatic_impairment=req.hepatic_impairment,
    )


@app.get("/api/logs")
def get_system_trace_logs(limit: int = 25) -> Dict:
    """Return recent ISO-8601 system trace logs including inputs, sources, model, outputs, and latency breakdown."""
    traces = read_recent_traces(limit=limit)
    return {"count": len(traces), "traces": traces}


@app.get("/api/drug/{drug_name}")
def get_or_index_drug(drug_name: str) -> Dict:
    """Normalize a drug via RxNorm and inspect its stored openFDA / DailyMed / WHO EML chunks."""
    return vector_store.ensure_drug_indexed(drug_name)


@app.post("/api/pharmacist/verify")
def verify_chunk(req: PharmacistVerifyRequest) -> Dict:
    """Spot-check endpoint for a clinical pharmacist to mark a stored knowledge chunk as verified."""
    updated = vector_store.mark_pharmacist_verified(
        chunk_id=req.chunk_id,
        verified=req.verified,
        notes=req.notes,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Chunk '{req.chunk_id}' not found.")
    iso_logger.info(f"PHARMACIST_VERIFY | chunk_id={req.chunk_id} | verified={req.verified} | notes={req.notes}")
    return {"status": "verified", "chunk_id": req.chunk_id}


@app.post("/api/seed")
def seed_knowledge_base(req: SeedRequest) -> Dict:
    """Pre-load a batch of common hospital formulary drugs into the Vector DB."""
    drugs_to_seed = req.drugs if req.drugs else PRIORITY_STARTER_DRUGS[: req.count]
    return seed_formulary_drugs(drug_list=drugs_to_seed, limit=len(drugs_to_seed))


if __name__ == "__main__":
    import uvicorn
    from config import CDSS_HOST, CDSS_PORT

    uvicorn.run("app:app", host=CDSS_HOST, port=CDSS_PORT, reload=False)
