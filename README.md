# Clinical Decision Support System (CDSS)

A local, privacy-preserving **Medication Safety & Clinical Decision Support System (CDSS)** powered by **RxNorm**, **openFDA**, **DailyMed**, **WHO Essential Medicines List (EML)**, a **Local Vector Database**, and your local **Ollama (`qwen3:8b`)** LLM.

---

## 1. How to Run the System

### Prerequisites
1. **Python Virtual Environment (`venv`)**: Already created in `./venv`.
2. **Local Ollama Running**: Make sure Ollama is running locally with `qwen3:8b` (or `qwen3-vl:4b`):
   ```powershell
   ollama list
   # If Ollama is not running, start it with:
   ollama serve
   ```

### Option A: One-Click Run (Windows Batch File)
Double-click **`run_cdss.bat`** in the project folder, or run it in PowerShell/CMD:
```powershell
.\run_cdss.bat
```

### Option B: Run via Terminal (`venv`)
```powershell
# 1. Activate the virtual environment (or call venv python directly)
.\venv\Scripts\Activate.ps1

# 2. (Optional) Pre-seed additional drugs from the 70-drug hospital formulary
python pipeline\seed_formulary.py 20

# 3. Start the FastAPI web server
python app.py
```

### Open the Web Interface
Once the server starts, open your browser to:
👉 **http://127.0.0.1:8000**

---

## 2. How the Knowledge & Evaluation Pipeline Works

```text
User Input (Patient ID, Name, Allergies/Conditions, Medications + Doses)
   │
   ▼
1. RxNorm REST API (pipeline/rxnorm.py)
   ├─ Normalizes Brand / International / Generic names → Standard Generic + RxCUI
   │  (e.g., "Brufen", "Advil", "Nurofen" → "ibuprofen" [RxCUI: 5640])
   │  (e.g., "Panadol", "Paracetamol", "Tylenol" → "acetaminophen" [RxCUI: 161])
   └─ Deterministic Duplicate Ingredient Check (flags same RxCUI across different brands)
   │
   ▼
2. Local Vector Store + Live Auto-Fetcher (pipeline/vector_store.py & label_fetcher.py)
   ├─ Checks if drug chunks exist in Data/vector_store.sqlite & Data/chunks/knowledge_chunks.jsonl
   ├─ If not cached yet, automatically fetches live from:
   │    • Primary: openFDA Drug Label API (indications, dosage, contraindications, interactions, warnings, pregnancy)
   │    • Fallback: DailyMed SPL XML API (LOINC section parser via setid)
   │    • Reference: WHO Essential Medicines List (Data/who_eml_core.json + PDF parser)
   └─ Saves chunks with full audit metadata (drug, rxcui, section, text, source, label_version_date, retrieved_on)
   │
   ▼
3. Local Qwen LLM Reasoning Engine (engine/cdss_evaluator.py)
   ├─ Retrieves targeted RAG chunks for each prescribed medication
   ├─ Cross-matches drug-drug interactions, allergies, contraindications, and dosing limits
   └─ Queries local Ollama (qwen3:8b) → Returns structured JSON:
        • 🟢 ALL_RIGHT   (Safe prescription, normal doses, no interactions/allergies)
        • 🟡 SUGGESTION  (Minor/moderate interaction, monitoring needed, or dose advice)
        • 🔴 ALERT       (High-risk drug interaction, allergy match, brand duplicate, or overdose)
```

---

## 3. Capabilities & Limitations Summary (Drug Interactions, Allergies, Dosing)

| Feature | Supported? | How It Works & Coverage | Known Limitations / Clinical Notes |
| :--- | :---: | :--- | :--- |
| **Any FDA / DailyMed Drug Lookup** | ✅ **Yes (Auto-Indexed Live)** | Even if a drug is not pre-loaded in `Data/vector_store.sqlite`, typing it in the UI automatically queries **RxNorm → openFDA → DailyMed** in real time, caches the chunks into the Vector DB, and evaluates it. | Drugs/brands registered *only* in Sri Lanka/UK and not in the US FDA database need an alias in `LOCAL_BRAND_AND_INN_ALIASES` (`pipeline/rxnorm.py`) or a local NMRA formulary file. Over 70 common Sri Lankan/UK brand & INN aliases (*Panadol*, *Brufen*, *Augmentin*, *Ecosprin*, *Losec*, *Plavix*, *Lasix*, *Salbutamol*, etc.) are already built in. |
| **Brand Duplicate Detection** | ✅ **Yes (100% Deterministic + LLM)** | RxNorm maps all brands to their base ingredient RxCUI. Entering *Brufen* + *Advil* (or *Panadol* + *Tylenol*) immediately flags a `DUPLICATE_THERAPY` alert. | Multi-ingredient fixed-dose combinations with non-US brand names should be entered by generic ingredients or added to the alias table. |
| **Drug–Drug Interactions** | ✅ **Yes (Narrative Label RAG + Class Matching)** | Pulls official `drug_interactions` and `warnings` sections from openFDA/DailyMed, cross-searches both specific generic names and pharmacologic classes (NSAIDs, anticoagulants, CYP2C19/CYP3A4 inhibitors, ACEi/ARBs, potassium-sparing diuretics, statins, macrolides, fluoroquinolones), and evaluates with `qwen3:8b`. | **Important**: As noted in the architecture spec, FDA labels describe interactions in *narrative text* rather than a structured pair-table. While major and well-known interactions (e.g., *Warfarin + Ibuprofen*, *Clopidogrel + Omeprazole*, *Simvastatin + Amlodipine/Clarithromycin*, *Enalapril + Spironolactone*) are caught reliably, obscure pair interactions that FDA labels describe only by enzyme pathway require a dedicated structured interaction DB (e.g., DDInter / First Databank / Lexicomp) + pharmacist verification for production hospital use. |
| **Drug Allergies & Cross-Reactivity** | ✅ **Yes** | Enter known allergies in the **"Allergies / Conditions / Clinical Notes"** field (e.g., `Allergic to Penicillin`, `NSAID allergy`, `Sulfa allergy`). The system checks FDA `contraindications` and `warnings` sections alongside Qwen's pharmacologic class knowledge (e.g., flagging *Amoxicillin* / *Augmentin* in a patient with a *Penicillin* allergy). | Requires the clinician to type the known allergy into the **Allergies / Conditions / Clinical Notes** input box (or integrate with your hospital EHR allergy table). |
| **Patient Conditions & Pregnancy Contraindications** | ✅ **Yes** | Enter patient conditions in the notes box (e.g., `Pregnant 3rd trimester`, `CKD eGFR 25`, `Active peptic ulcer`). Checked against FDA `contraindications`, `warnings`, `pregnancy`, and `WHO EML` notes. | Dependent on the clinical context entered in the notes field. |
| **Dosage & Overdose Checking** | ✅ **Yes** | Compares prescribed dose and frequency (e.g., `400mg TID`, `1500mg q4h`) against openFDA `dosage_and_administration` and `WHO EML` typical adult doses. | Pediatric weight-based (`mg/kg`) calculations require the patient's weight/age to be written in the notes field. |
| **Pharmacist Spot-Checking** | ✅ **Yes** | Every chunk is saved in human-readable JSONL at `Data/chunks/knowledge_chunks.jsonl` and can be inspected & marked **"✅ Pharmacist Verified"** in Tab 2 of the Web UI. | Recommended workflow: Have a clinical pharmacist review and verify the core 50–100 hospital formulary drugs in Tab 2. |

---

## 4. Project Structure

```text
d:\Projects\11 Future\Avenirit\Work\CDSS\
├── app.py                        # FastAPI server & REST API endpoints
├── config.py                     # API URLs, file paths, and Ollama model config
├── requirements.txt              # Python package dependencies
├── run_cdss.bat                  # One-click Windows startup script
├── README.md                     # Documentation & clinical capability guide
├── pipeline/
│   ├── rxnorm.py                 # RxNorm REST API normalizer + UK/SL brand/INN aliases
│   ├── label_fetcher.py          # openFDA JSON + DailyMed SPL XML fallback fetcher
│   ├── who_eml.py                # WHO Essential Medicines reference & PDF parser
│   ├── vector_store.py           # Local SQLite+NumPy Vector DB & JSONL chunk manager
│   └── seed_formulary.py         # CLI script to pre-load hospital formulary drugs
├── engine/
│   └── cdss_evaluator.py         # RAG prompt builder + Local Ollama (qwen3:8b) evaluator
├── static/
│   └── index.html                # Clinical check frontend + Pharmacist Spot-Check UI
└── Data/
    ├── formulary_seed.json       # List of 70 common hospital formulary drugs
    ├── who_eml_core.json         # WHO EML reference data & typical adult doses
    ├── vector_store.sqlite       # Persistent local vector database
    ├── evaluation_history.jsonl  # Audit log of all evaluated prescriptions
    └── chunks/
        └── knowledge_chunks.jsonl # Raw JSON chunks for pharmacist spot-checking
```

