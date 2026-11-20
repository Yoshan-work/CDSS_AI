import re
import logging
from typing import Dict, Any, List, Optional
import requests
from config import RXNORM_BASE_URL

logger = logging.getLogger(__name__)

# Common Sri Lanka / UK / International brand & INN aliases mapped to US RxNorm generic names
# This ensures brands not registered in the US (like Panadol, Brufen, Ecosprin, Salbutamol)
# still resolve cleanly to their exact RxNorm RxCUI and openFDA generic name.
LOCAL_BRAND_AND_INN_ALIASES: Dict[str, str] = {
    "panadol": "acetaminophen",
    "paracetamol": "acetaminophen",
    "calpol": "acetaminophen",
    "tylenol": "acetaminophen",
    "brufen": "ibuprofen",
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "nurofen": "ibuprofen",
    "voltaren": "diclofenac",
    "voveran": "diclofenac",
    "cataflam": "diclofenac",
    "ecosprin": "aspirin",
    "disprin": "aspirin",
    "aspro": "aspirin",
    "acetylsalicylic acid": "aspirin",
    "plavix": "clopidogrel",
    "clopilet": "clopidogrel",
    "coumadin": "warfarin",
    "marevan": "warfarin",
    "xarelto": "rivaroxaban",
    "eliquis": "apixaban",
    "clexane": "enoxaparin",
    "augmentin": "amoxicillin",
    "amoxil": "amoxicillin",
    "zithromax": "azithromycin",
    "azee": "azithromycin",
    "cipro": "ciprofloxacin",
    "ciplox": "ciprofloxacin",
    "flagyl": "metronidazole",
    "metrogyl": "metronidazole",
    "rocephin": "ceftriaxone",
    "keflex": "cephalexin",
    "klacid": "clarithromycin",
    "biaxin": "clarithromycin",
    "diflucan": "fluconazole",
    "glucophage": "metformin",
    "amaryl": "glimepiride",
    "diamicron": "gliclazide",
    "januvia": "sitagliptin",
    "jardiance": "empagliflozin",
    "norvasc": "amlodipine",
    "amlong": "amlodipine",
    "cozaar": "losartan",
    "losacar": "losartan",
    "vasotec": "enalapril",
    "envas": "enalapril",
    "zestril": "lisinopril",
    "diovan": "valsartan",
    "tenormin": "atenolol",
    "betaloc": "metoprolol",
    "lopressor": "metoprolol",
    "concor": "bisoprolol",
    "lasix": "furosemide",
    "frusemide": "furosemide",
    "aldactone": "spironolactone",
    "lipitor": "atorvastatin",
    "atorva": "atorvastatin",
    "crestor": "rosuvastatin",
    "zocor": "simvastatin",
    "lanoxin": "digoxin",
    "cordarone": "amiodarone",
    "losec": "omeprazole",
    "prilosec": "omeprazole",
    "omez": "omeprazole",
    "pantocid": "pantoprazole",
    "protonix": "pantoprazole",
    "nexium": "esomeprazole",
    "pepcid": "famotidine",
    "zofran": "ondansetron",
    "emeset": "ondansetron",
    "maxolon": "metoclopramide",
    "perinorm": "metoclopramide",
    "salbutamol": "albuterol",
    "ventolin": "albuterol",
    "astHALIN".lower(): "albuterol",
    "singulair": "montelukast",
    "wysolone": "prednisolone",
    "decadron": "dexamethasone",
    "eltroxin": "levothyroxine",
    "synthroid": "levothyroxine",
    "thyronorm": "levothyroxine",
    "zoloft": "sertraline",
    "prozac": "fluoxetine",
    "elavil": "amitriptyline",
    "valium": "diazepam",
    "ativan": "lorazepam",
    "tegretol": "carbamazepine",
    "epilim": "valproic acid",
    "depakote": "valproic acid",
    "keppra": "levetiracetam",
    "zyloric": "allopurinol",
    "ultram": "tramadol",
}

# In-memory cache to avoid redundant network hits for the same drug name
_RXNORM_CACHE: Dict[str, Dict[str, Any]] = {}


def clean_drug_query(raw_name: str) -> str:
    """Strip dosage numbers/units accidentally typed into the drug name field."""
    name = raw_name.strip().lower()
    # Remove trailing strengths like "400mg", "500 mg", "5mg/5ml", "tablets", "caps"
    name = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|%)\b.*", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\b(tablet|tablets|tab|tabs|capsule|capsules|cap|caps|syrup|injection|inj|oral)\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name).strip()
    return name or raw_name.strip().lower()


def normalize_drug_name(drug_name: str, timeout: int = 8) -> Dict[str, Any]:
    """
    Normalize a brand or generic drug name using the NLM RxNorm REST API.
    1. Resolves local/international aliases if applicable.
    2. Queries /rxcui.json?name={name}&search=2
    3. Falls back to /approximateTerm.json if needed.
    4. Queries /rxcui/{id}/allrelated.json to extract the base generic ingredient (IN),
       RxCUI, brand names (BN), and related ingredients.
    """
    cleaned_input = clean_drug_query(drug_name)
    if cleaned_input in _RXNORM_CACHE:
        cached = dict(_RXNORM_CACHE[cleaned_input])
        cached["entered_name"] = drug_name
        return cached

    # Check if we have a known INN / Sri Lankan / UK alias
    lookup_term = LOCAL_BRAND_AND_INN_ALIASES.get(cleaned_input, cleaned_input)

    rxcui: Optional[str] = None
    generic_name: str = lookup_term
    ingredients: List[str] = []
    brands: List[str] = []
    resolution_method = "RxNorm REST API"

    session = requests.Session()
    try:
        # Step 1: Try exact/normalized lookup on lookup_term (and original input if different)
        for candidate in ([cleaned_input, lookup_term] if cleaned_input != lookup_term else [lookup_term]):
            resp = session.get(
                f"{RXNORM_BASE_URL}/rxcui.json",
                params={"name": candidate, "search": 2},
                timeout=timeout,
            )
            if resp.ok:
                data = resp.json()
                rxnorm_ids = data.get("idGroup", {}).get("rxnormId", [])
                if rxnorm_ids:
                    rxcui = str(rxnorm_ids[0])
                    break

        # Step 2: Fallback to approximateTerm if exact match not found
        if not rxcui:
            resp = session.get(
                f"{RXNORM_BASE_URL}/approximateTerm.json",
                params={"term": lookup_term, "maxEntries": 3},
                timeout=timeout,
            )
            if resp.ok:
                candidates = (
                    resp.json()
                    .get("approximateGroup", {})
                    .get("candidate", [])
                )
                if candidates:
                    rxcui = str(candidates[0].get("rxcui"))
                    resolution_method = "RxNorm Approximate Match"

        # Step 3: Fetch /rxcui/{id}/allrelated.json to normalize Brand -> Ingredient (IN)
        if rxcui:
            rel_resp = session.get(
                f"{RXNORM_BASE_URL}/rxcui/{rxcui}/allrelated.json",
                timeout=timeout,
            )
            if rel_resp.ok:
                concept_groups = (
                    rel_resp.json()
                    .get("allRelatedGroup", {})
                    .get("conceptGroup", [])
                )
                in_concepts = []
                min_concepts = []
                bn_concepts = []

                for group in concept_groups:
                    tty = group.get("tty")
                    props = group.get("conceptProperties", [])
                    if tty == "IN":
                        in_concepts.extend(props)
                    elif tty == "MIN":
                        min_concepts.extend(props)
                    elif tty == "BN":
                        bn_concepts.extend(props)

                if in_concepts:
                    # Normalize RxCUI to the primary Ingredient (IN) RxCUI so brand and generic share the exact same ID!
                    primary_in = in_concepts[0]
                    rxcui = str(primary_in.get("rxcui", rxcui))
                    generic_name = str(primary_in.get("name", lookup_term)).lower()
                    ingredients = [str(c.get("name", "")).lower() for c in in_concepts if c.get("name")]
                elif min_concepts:
                    primary_min = min_concepts[0]
                    rxcui = str(primary_min.get("rxcui", rxcui))
                    generic_name = str(primary_min.get("name", lookup_term)).lower()
                    ingredients = [s.strip().lower() for s in generic_name.split("/")]

                brands = [str(b.get("name", "")) for b in bn_concepts[:10] if b.get("name")]

    except Exception as exc:
        logger.warning(f"RxNorm lookup warning for '{drug_name}': {exc}")
        resolution_method = "Local Fallback Alias"

    if not ingredients:
        ingredients = [generic_name]

    result = {
        "entered_name": drug_name,
        "cleaned_query": cleaned_input,
        "generic_name": generic_name,
        "rxcui": rxcui or "UNKNOWN",
        "ingredients": ingredients,
        "sample_brands": brands[:6],
        "resolution_source": resolution_method,
    }
    _RXNORM_CACHE[cleaned_input] = result
    return dict(result)

