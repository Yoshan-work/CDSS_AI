# CDSS — Complete System Documentation

## 1. Full Application Summary

This application is a **local Clinical Decision Support System (CDSS)** for **medication safety checking**. It takes a patient's ID, name, optional clinical notes (allergies, conditions, pregnancy, renal status), and a list of prescribed medications with doses, and evaluates them using a **Hybrid RAG (Retrieval-Augmented Generation)** pipeline powered by your local **Ollama (`qwen3:8b`)** LLM.

### Key Capabilities
- **Brand & Generic Normalization:** Resolves brand names and international/Sri Lankan INN names (e.g., *Brufen*, *Advil* → `ibuprofen` [`RxCUI: 5640`]; *Panadol*, *Paracetamol* → `acetaminophen` [`RxCUI: 161`]).
- **Duplicate Active Ingredient Detection:** Deterministically flags when two different brand names share the same normalized `RxCUI` or generic ingredient.
- **Drug–Drug Interaction Checking:** Cross-references official `drug_interactions`, `warnings`, and `contraindications` label sections across all prescribed drugs.
- **Dose & Overdose Checking:** Compares entered doses/frequencies against `dosage_and_administration` label text and WHO EML typical adult doses.
- **Allergy & Condition Screening:** Checks patient notes (e.g., *Penicillin allergy*, *CKD*, *Pregnancy*, *Peptic ulcer*) against contraindications, warnings, pregnancy sections, and pharmacologic class cross-reactivity.
- **3-Tier Clinical Output:**
  - `ALL_RIGHT` (🟢): No critical issues detected.
  - `SUGGESTION` (🟡): Minor/moderate monitoring or administration advice.
  - `ALERT` (🔴): High-risk interaction, duplicate therapy, allergy match, overdose, or contraindication.
- **Pharmacist Spot-Checking:** Every retrieved knowledge chunk is saved in a human-readable JSONL file (`Data/chunks/knowledge_chunks.jsonl`) and can be inspected and marked as verified in the UI.

---

## 2. Full End-to-End Flow

```text
1. User Input (static/index.html)
   └─ Patient ID, Patient Name, Allergies/Clinical Notes, Medications [{name, dose}]
   └─ POST /api/check -> app.py (check_prescription)
         │
         ▼
2. Drug Normalization (pipeline/rxnorm.py -> normalize_drug_name)
   ├─ clean_drug_query(): strips strengths/dosage forms ("400mg", "tablets")
   ├─ Checks LOCAL_BRAND_AND_INN_ALIASES (70+ Sri Lankan/UK/US brand & INN aliases)
   ├─ Queries NLM RxNorm REST API:
   │    • /REST/rxcui.json?name={candidate}&search=2
   │    • Fallback: /REST/approximateTerm.json?term={term}&maxEntries=3
   │    • /REST/rxcui/{id}/allrelated.json -> extracts primary Ingredient (IN) RxCUI & name
   └─ Caches result in memory (_RXNORM_CACHE)
         │
         ▼
3. Vector DB Lookup & On-Demand Knowledge Fetching (pipeline/vector_store.py -> ensure_drug_indexed)
   ├─ Checks local SQLite Vector DB (Data/vector_store.sqlite) by generic_name / rxcui
   ├─ IF NOT FOUND IN LOCAL DB:
   │    ├─ Calls fetch_drug_label_chunks() (pipeline/label_fetcher.py):
   │    │    • Primary: openFDA API (https://api.fda.gov/drug/label.json)
   │    │      Extracts: indications_and_usage, contraindications, drug_interactions,
   │    │                dosage_and_administration, warnings, pregnancy
   │    │    • Fallback: DailyMed SPL XML API (if <3 sections or critical sections missing)
   │    │      Searches /spls.json -> gets setid -> fetches /spls/{setid}.xml -> parses LOINC codes
   │    ├─ Calls build_who_eml_chunk() (pipeline/who_eml.py -> Data/who_eml_core.json)
   │    └─ Computes 1024-dim local embeddings (compute_text_embedding) & saves to:
   │         • Data/vector_store.sqlite
   │         • Data/chunks/knowledge_chunks.jsonl
   └─ Returns stored chunks + WHO EML metadata
         │
         ▼
4. Deterministic & RAG Context Assembly (engine/cdss_evaluator.py -> evaluate_prescription)
   ├─ Pre-check: Groups medications by RxCUI -> flags DUPLICATE_THERAPY if >1 drug shares RxCUI
   ├─ Builds focused RAG snippets using extract_focused_snippet() + DRUG_CLASS_KEYWORDS
   └─ Sends system rules + patient context + RAG evidence to Local Ollama (/api/chat)
         │
         ▼
5. Local Qwen LLM Evaluation (engine/cdss_evaluator.py -> _call_qwen_llm)
   ├─ Model: qwen3:8b (auto-detected via /api/tags, temperature=0.1, format="json", think=False)
   ├─ Parses JSON output via parse_llm_json()
   ├─ Merges any deterministic duplicate warnings
   ├─ Logs summary to Data/evaluation_history.jsonl
   └─ Returns structured JSON to UI -> displays Banner, Alerts, Suggestions, Table, & Citations
```

---

## 3. Online vs. Offline Mode & Limitations

The system is a **Hybrid Online/Offline** architecture:

### What Runs 100% Offline (Local Machine)
- **Web Server & UI:** FastAPI (`app.py`) + HTML/JS (`static/index.html`).
- **LLM Reasoning:** Local Ollama (`http://localhost:11434`) running `qwen3:8b` (no cloud LLM API keys or internet needed for AI inference).
- **Vector Database & Embeddings:** `Data/vector_store.sqlite` uses deterministic local NumPy feature-hashing embeddings (`compute_text_embedding`, 1024 dimensions)—zero external embedding API calls.
- **Cached Drugs & WHO EML:** Any drug already indexed in `Data/vector_store.sqlite` + `Data/who_eml_core.json` works completely offline.

### What Requires Internet (Online)
- **First-Time Lookup of Uncached Drugs:** When you enter a drug that has **never** been indexed into `Data/vector_store.sqlite` before, `normalize_drug_name()` calls **RxNorm** (`rxnav.nlm.nih.gov`) and `fetch_drug_label_chunks()` calls **openFDA** (`api.fda.gov`) / **DailyMed** (`dailymed.nlm.nih.gov`).
- **Once fetched once**, the drug is permanently cached in `Data/vector_store.sqlite` and `Data/chunks/knowledge_chunks.jsonl`, so subsequent checks for that drug work **offline** (note: if internet is down during a cached drug check, `normalize_drug_name` will fall back to `LOCAL_BRAND_AND_INN_ALIASES` or the raw cleaned drug name and still retrieve cached chunks by generic name).

### Key Limitations
1. **Narrative Interaction Text (FDA Labels):** openFDA and DailyMed store drug interactions as narrative prose rather than a structured pair matrix. While `DRUG_CLASS_KEYWORDS` and `qwen3:8b` bridge many class-level interactions, safety-critical hospital deployments should supplement this with a structured interaction database (e.g., DDInter, First Databank, or Lexicomp) and pharmacist review.
2. **US-Centric Labels vs. Local Formulary:** openFDA and DailyMed are US FDA labels. Brands/strengths registered only in Sri Lanka (NMRA) or the UK (BNF) must either be entered by generic name or mapped in `LOCAL_BRAND_AND_INN_ALIASES` (`pipeline/rxnorm.py`).
3. **Unseeded Drugs Offline:** If the machine is completely offline and a drug was never pre-seeded (via `pipeline/seed_formulary.py` or a prior online check), the system cannot fetch its FDA label and will evaluate using only the LLM's base knowledge + fallback warning.
4. **No Patient EHR Integration Yet:** Allergies, pregnancy, renal function (eGFR), and weight must be typed manually into the *Allergies / Conditions / Clinical Notes* box.
5. **Rate Limits (openFDA):** Without an `OPENFDA_API_KEY` environment variable, openFDA limits unauthenticated requests (240 req/min, 1,000/day). Bulk pre-seeding 70+ drugs is best done once and cached locally.

---

## 4. APIs & Knowledge Bases Used

### External APIs
| Source | Endpoint Used | Auth / Key | Purpose |
| :--- | :--- | :--- | :--- |
| **NLM RxNorm REST API** | `https://rxnav.nlm.nih.gov/REST/rxcui.json`<br>`/REST/approximateTerm.json`<br>`/REST/rxcui/{id}/allrelated.json` | Free, No Key | Brand → Generic Ingredient (`IN`) → `RxCUI` normalization. |
| **openFDA Drug Label API** | `https://api.fda.gov/drug/label.json` | Free (Optional `OPENFDA_API_KEY` env var) | Primary source for 6 label sections (`indications_and_usage`, `contraindications`, `drug_interactions`, `dosage_and_administration`, `warnings`, `pregnancy`). |
| **DailyMed REST API** | `https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json`<br>`/spls/{setid}.xml` | Free, No Key | Fallback SPL XML label parser using HL7 LOINC section codes (`34067-9`, `34070-3`, `34073-7`, `34068-7`, `34071-1`, `43685-7`, `34066-1`, `42228-7`, `34080-2`). |
| **Local Ollama API** | `http://localhost:11434/api/tags`<br>`http://localhost:11434/api/chat` | Localhost | Runs `qwen3:8b` (with fallback list in `config.py`). |

### Internal FastAPI Endpoints (`app.py`)
- `GET /` — Serves `static/index.html`.
- `GET /api/status` — Returns active Ollama model and Vector DB counts (`total_drugs`, `total_chunks`, `verified_chunks`, `indexed_drugs`).
- `POST /api/check` — Runs the full `CDSSEvaluator.evaluate_prescription()` pipeline.
- `GET /api/drug/{drug_name}` — Normalizes and inspects/indexes a single drug's chunks.
- `POST /api/pharmacist/verify` — Marks a `chunk_id` as pharmacist-verified in SQLite.
- `POST /api/seed` — Pre-loads starter formulary drugs from `PRIORITY_STARTER_DRUGS`.

### Local Knowledge Base Files (`Data/`)
- `Data/formulary_seed.json` — 70 common hospital formulary drugs.
- `Data/who_eml_core.json` — Structured WHO Essential Medicines List (23rd List reference) entries with EML status, indications, typical adult doses, and clinical notes.
- `Data/vector_store.sqlite` — SQLite table (`chunks`) storing text, metadata, pharmacist verification flags, and 1024-dim float32 embedding BLOBs.
- `Data/chunks/knowledge_chunks.jsonl` — Append-only JSONL mirror of ingested chunks for pharmacist spot-checking.
- `Data/evaluation_history.jsonl` — Audit log of evaluated prescriptions.
- `pipeline/who_eml.py` also includes `parse_who_eml_pdf()` (using `pdfplumber`) to ingest full WHO EML or national formulary PDFs.

---

## 5. Code Quality & Architecture Assessment

### Strengths
- **Clean Separation of Concerns:** Configuration (`config.py`), ingestion/normalization (`pipeline/`), LLM reasoning (`engine/`), API layer (`app.py`), and frontend (`static/index.html`) are cleanly separated.
- **Resilience & Fallbacks:**
  - RxNorm falls back from exact match → `approximateTerm` → `LOCAL_BRAND_AND_INN_ALIASES`.
  - Label fetching falls back from `openFDA` → `DailyMed` SPL XML when critical sections are missing.
  - LLM output parsing (`parse_llm_json`) strips `<think>` tags and markdown fences safely, and `_call_qwen_llm` degrades gracefully if Ollama is unreachable.
- **Auditability:** Every chunk stores `source`, `label_version_date`, `retrieved_on`, and `pharmacist_verified` status, and every evaluation is logged to `Data/evaluation_history.jsonl`.
- **Deterministic Guardrails:** Duplicate active ingredient checks run deterministically in Python via `RxCUI` before the LLM is even called.

### Areas for Future Improvement (Technical Debt / Next Steps)
1. **Unit & Integration Test Suite:** Add `pytest` tests mocking RxNorm/openFDA/Ollama responses to verify edge cases (combination drugs, missing fields, malformed JSON).
2. **Thread-Safe / Async HTTP & DB Pooling:** `app.py` uses synchronous `requests.Session()` and `sqlite3` calls inside standard `def` FastAPI routes (which FastAPI runs in a threadpool). Moving to `httpx.AsyncClient` and connection pooling would improve concurrency under heavy multi-user hospital load.
3. **Chunk Deduplication in JSONL:** `Data/vector_store.sqlite` uses `INSERT OR REPLACE` on `chunk_id`, whereas `Data/chunks/knowledge_chunks.jsonl` only appends new `chunk_id`s; adding a refresh/overwrite helper would keep JSONL in sync if labels are re-fetched later.
4. **Structured Interaction DB Integration:** Integrating a structured interaction dataset (such as DDInter from `Sources/sources_1.txt`) alongside narrative FDA chunks would eliminate reliance on narrative text matching for rare drug pairs.
5. **Pre-Seeding All 70 Formulary Drugs:** Running `python pipeline/seed_formulary.py` for the full `Data/formulary_seed.json` list will make all 70 core drugs available 100% offline.

