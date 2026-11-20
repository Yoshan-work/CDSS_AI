import logging
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine.cdss_evaluator import CDSSEvaluator, get_active_ollama_model
from pipeline.vector_store import CDSSVectorStore
from pipeline.seed_formulary import seed_formulary_drugs, PRIORITY_STARTER_DRUGS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CDSS — Clinical Decision Support System (RxNorm + openFDA + DailyMed + WHO EML + Local Qwen LLM)",
    version="1.0.0",
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

vector_store = CDSSVectorStore()
evaluator = CDSSEvaluator(vector_store=vector_store)


class MedicationItem(BaseModel):
    name: str = Field(..., description="Brand or generic medication name (e.g., Brufen, Warfarin, Panadol)")
    dose: str = Field(..., description="Dose and frequency (e.g., 400mg TID, 5mg OD)")


class PrescriptionCheckRequest(BaseModel):
    patient_id: str = Field(..., description="Patient Identifier")
    patient_name: str = Field(..., description="Patient Full Name")
    patient_notes: Optional[str] = Field(
        default="",
        description="Optional clinical notes, age, conditions, or allergies (e.g., Pregnant, CKD, Peptic Ulcer)",
    )
    medications: List[MedicationItem]


class PharmacistVerifyRequest(BaseModel):
    chunk_id: str
    verified: bool = True
    notes: str = ""


class SeedRequest(BaseModel):
    drugs: Optional[List[str]] = None
    count: int = 10


@app.get("/")
def serve_index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status")
def system_status():
    stats = vector_store.get_stats()
    active_model = get_active_ollama_model()
    return {
        "status": "ok",
        "ollama_model": active_model,
        "knowledge_base": stats,
    }


@app.post("/api/check")
def check_prescription(req: PrescriptionCheckRequest):
    valid_meds = [
        {"name": m.name.strip(), "dose": m.dose.strip()}
        for m in req.medications
        if m.name.strip()
    ]
    if not valid_meds:
        raise HTTPException(status_code=400, detail="Please enter at least one medication.")

    result = evaluator.evaluate_prescription(
        patient_id=req.patient_id.strip() or "N/A",
        patient_name=req.patient_name.strip() or "Unknown Patient",
        medications=valid_meds,
        patient_notes=(req.patient_notes or "").strip(),
    )
    return result


@app.get("/api/drug/{drug_name}")
def get_or_index_drug(drug_name: str):
    """Normalize a drug via RxNorm and inspect its stored openFDA / DailyMed / WHO EML chunks."""
    return vector_store.ensure_drug_indexed(drug_name)


@app.post("/api/pharmacist/verify")
def verify_chunk(req: PharmacistVerifyRequest):
    """Spot-check endpoint for a clinical pharmacist to mark a stored knowledge chunk as verified."""
    updated = vector_store.mark_pharmacist_verified(
        chunk_id=req.chunk_id,
        verified=req.verified,
        notes=req.notes,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Chunk '{req.chunk_id}' not found.")
    return {"status": "verified", "chunk_id": req.chunk_id}


@app.post("/api/seed")
def seed_knowledge_base(req: SeedRequest):
    """Pre-load a batch of common hospital formulary drugs into the Vector DB."""
    drugs_to_seed = req.drugs if req.drugs else PRIORITY_STARTER_DRUGS[: req.count]
    return seed_formulary_drugs(drug_list=drugs_to_seed, limit=len(drugs_to_seed))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)

