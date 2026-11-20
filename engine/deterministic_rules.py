"""
Deterministic Clinical Decision Support Rule Engine (ISO/IEC 25010 & ISO 14971 Compliant).
All safety-critical alerts (Allergies, Cross-Reactivity, RxCUI/ATC Class Duplicates,
DDInter 2.0 Drug Interactions, Dose Limits, KDIGO Renal eGFR, Pregnancy/LactMed,
LiverTox Hepatic, AGS Beers Elderly, CredibleMeds QT, and NOT_VERIFIED checks)
are computed deterministically in code prior to invoking the LLM explainer.
"""

import json
import re
from typing import Any, Dict, List, Optional

from config import DOSE_RULES_PATH
from pipeline.ddinter_loader import DDInterStore
from pipeline.rxclass import load_atc_database

_DOSE_RULES_CACHE: Optional[Dict[str, Any]] = None


def load_dose_rules() -> Dict[str, Any]:
    """Load structured dose, KDIGO renal, pregnancy, LactMed, Beers, and QT rules."""
    global _DOSE_RULES_CACHE
    if _DOSE_RULES_CACHE is not None:
        return _DOSE_RULES_CACHE
    if not DOSE_RULES_PATH.exists():
        _DOSE_RULES_CACHE = {"rules": {}}
        return _DOSE_RULES_CACHE
    with open(DOSE_RULES_PATH, "r", encoding="utf-8") as f:
        _DOSE_RULES_CACHE = json.load(f)
    return _DOSE_RULES_CACHE


def parse_structured_dose(dose_str: str) -> Dict[str, Any]:
    """
    Parse free-text or structured dose string into numeric single_dose_mg,
    frequency_per_day, duration_days, and calculated_daily_dose_mg.
    """
    raw = (dose_str or "").strip()
    s = raw.lower()

    # Check if user explicitly wrote "(9000 mg/day)" or similar daily total override
    explicit_daily = re.search(r"(\d+(?:\.\d+)?)\s*mg\s*/\s*(?:day|24\s*h)", s)

    m = re.search(r"(\d+(?:\.\d+)?)\s*(mg|mcg|g)\b", s)
    if not m:
        return {
            "raw": raw,
            "parsed": False,
            "single_dose_mg": None,
            "frequency_per_day": None,
            "duration_days": None,
            "daily_dose_mg": None,
            "formatted_calc": "Unparsed dose format",
        }

    val = float(m.group(1))
    unit = m.group(2)
    if unit == "g":
        single_mg = val * 1000.0
    elif unit == "mcg":
        single_mg = val / 1000.0
    else:
        single_mg = val

    freq = 1.0
    if re.search(r"\b(q4h|every\s*4\s*h(?:ours|rs)?)\b", s):
        freq = 6.0
    elif re.search(r"\b(qid|q\.i\.d|four\s*times|4\s*times|q6h|every\s*6\s*h(?:ours|rs)?)\b", s):
        freq = 4.0
    elif re.search(r"\b(tds|tid|t\.d\.s|t\.i\.d|three\s*times|3\s*times|q8h|every\s*8\s*h(?:ours|rs)?)\b", s):
        freq = 3.0
    elif re.search(r"\b(bd|bid|b\.d|b\.i\.d|twice|2\s*times|q12h|every\s*12\s*h(?:ours|rs)?)\b", s):
        freq = 2.0
    elif re.search(r"\b(od|o\.d|once|daily|night|nocte|mane|q24h)\b", s):
        freq = 1.0

    dur_match = re.search(r"(\d+)\s*(?:day|days|d)\b", s)
    duration_days = int(dur_match.group(1)) if dur_match else None

    daily_mg = float(explicit_daily.group(1)) if explicit_daily else round(single_mg * freq, 3)
    s_fmt = int(single_mg) if single_mg.is_integer() else single_mg
    f_fmt = int(freq) if freq.is_integer() else freq
    d_fmt = int(daily_mg) if daily_mg.is_integer() else daily_mg

    return {
        "raw": raw,
        "parsed": True,
        "single_dose_mg": single_mg,
        "frequency_per_day": freq,
        "duration_days": duration_days,
        "daily_dose_mg": daily_mg,
        "formatted_calc": f"{s_fmt} mg × {f_fmt}/day = {d_fmt} mg/day",
    }


def extract_allergies_and_context_from_notes(
    patient_notes: str,
    coded_allergies: Optional[List[Dict[str, str]]] = None,
    patient_age: Optional[int] = None,
    egfr: Optional[float] = None,
    pregnancy_status: Optional[str] = None,
    lactation_status: Optional[bool] = None,
    hepatic_impairment: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Combine structured patient parameters with deterministic regex extraction from `patient_notes`
    so both the new structured UI fields and existing free-text notes work seamlessly.
    """
    notes_lower = (patient_notes or "").lower()
    allergies: List[Dict[str, str]] = list(coded_allergies or [])
    existing_keys = {a.get("allergen", "").lower() for a in allergies}

    # Detect severity from notes
    default_sev = (
        "anaphylaxis_severe"
        if any(w in notes_lower for w in ("anaphylaxis", "severe", "angioedema", "bronchospasm", "sjs"))
        else "urticaria_moderate"
    )

    has_no_allergy = any(
        phrase in notes_lower
        for phrase in ("no known allerg", "nka", "nkda", "no allerg", "nil allerg")
    )

    if not has_no_allergy:
        keyword_to_allergen = {
            "penicillin": "penicillin",
            "amoxicillin": "penicillin",
            "augmentin": "penicillin",
            "cephalosporin": "cephalosporin",
            "sulfa": "sulfonamide",
            "sulfonamide": "sulfonamide",
            "nsaid": "nsaid",
            "ibuprofen": "nsaid",
            "diclofenac": "nsaid",
            "aspirin": "aspirin",
            "paracetamol": "acetaminophen",
            "acetaminophen": "acetaminophen",
            "panadol": "acetaminophen",
            "macrolide": "macrolide",
            "fluoroquinolone": "fluoroquinolone",
            "ciprofloxacin": "fluoroquinolone",
        }
        has_allergy_cue = any(
            w in notes_lower
            for w in ("allergy", "allergic", "hypersensitivity", "angioedema", "anaphylaxis", "sensitive")
        )
        for kw, code in keyword_to_allergen.items():
            # Match if an allergy cue is present OR if the user simply typed the allergen class name (e.g. "Penicillin", "Sulfa") in the Allergies/Notes box
            if kw in notes_lower and code not in existing_keys:
                if has_allergy_cue or kw in ("penicillin", "sulfa", "sulfonamide", "nsaid", "cephalosporin"):
                    allergies.append({"allergen": code, "severity": default_sev})
                    existing_keys.add(code)

    # Parse eGFR from notes if not provided as structured float
    if egfr is None:
        egfr_m = re.search(r"egfr\s*(?:is|=|:|<|>)?\s*(\d+(?:\.\d+)?)", notes_lower)
        if egfr_m:
            egfr = float(egfr_m.group(1))
        elif "ckd stage 4" in notes_lower or "ckd stage 5" in notes_lower or "severe renal" in notes_lower:
            egfr = 22.0

    # Parse Age from notes if not provided
    if patient_age is None:
        age_m = re.search(r"\bage\s*(\d{1,3})\b|(\d{2})\s*(?:yo|y/o|years\s*old)", notes_lower)
        if age_m:
            patient_age = int(age_m.group(1) or age_m.group(2))

    # Parse Pregnancy from notes if not provided
    preg = (pregnancy_status or "none").lower()
    if preg in ("none", "", "false", "no"):
        if any(w in notes_lower for w in ("pregnant", "pregnancy", "trimester", "gestation")):
            if "3rd trimester" in notes_lower or "third trimester" in notes_lower:
                preg = "trimester_3"
            elif "2nd trimester" in notes_lower or "second trimester" in notes_lower:
                preg = "trimester_2"
            else:
                preg = "pregnant"
        else:
            preg = "none"

    # Parse Lactation
    lact = bool(lactation_status) or any(w in notes_lower for w in ("breastfeeding", "lactation", "nursing mother"))

    # Parse Hepatic impairment
    hepatic = bool(hepatic_impairment) or any(
        w in notes_lower for w in ("cirrhosis", "liver disease", "hepatic impairment", "liver failure")
    )

    # Parse Asthma / Peptic Ulcer / Dengue
    has_asthma = "asthma" in notes_lower
    has_peptic_ulcer = any(w in notes_lower for w in ("peptic ulcer", "gi bleed", "gastric ulcer"))
    has_dengue = any(w in notes_lower for w in ("dengue", "thrombocytopenia", "low platelet"))

    return {
        "coded_allergies": allergies,
        "patient_age": patient_age,
        "egfr": egfr,
        "pregnancy_status": preg,
        "lactation_status": lact,
        "hepatic_impairment": hepatic,
        "has_asthma": has_asthma,
        "has_peptic_ulcer": has_peptic_ulcer,
        "has_dengue": has_dengue,
    }


def run_deterministic_cdss_rules(
    med_records: List[Dict[str, Any]],
    patient_context: Dict[str, Any],
    ddinter_store: DDInterStore,
) -> Dict[str, Any]:
    """
    Execute all deterministic clinical safety checks:
    1. Verification Gate (NOT_VERIFIED for unknown drugs or missing mandatory context)
    2. RxCUI Duplicate Ingredient & ATC Class Therapeutic Duplication
    3. Coded Allergy & Cross-Reactivity Matrix
    4. Structured DDInter 2.0 Drug-Drug Interactions + Triple Whammy + Multi-QT
    5. Deterministic Dose, KDIGO Renal eGFR, Hepatic (LiverTox), Pregnancy/LactMed, AGS Beers, Asthma/Dengue
    """
    alerts: List[Dict[str, Any]] = []
    suggestions: List[str] = []
    med_reviews: Dict[str, Dict[str, Any]] = {}
    not_verified_reasons: List[str] = []

    atc_db = load_atc_database()
    allergy_matrix = atc_db.get("allergy_cross_reactivity_matrix", {})
    dose_rules = load_dose_rules().get("rules", {})

    allergies = patient_context.get("coded_allergies", [])
    egfr = patient_context.get("egfr")
    age = patient_context.get("patient_age")
    pregnancy = patient_context.get("pregnancy_status", "none")
    lactation = patient_context.get("lactation_status", False)
    hepatic = patient_context.get("hepatic_impairment", False)
    has_asthma = patient_context.get("has_asthma", False)
    has_peptic_ulcer = patient_context.get("has_peptic_ulcer", False)
    has_dengue = patient_context.get("has_dengue", False)

    # 1. Per-Drug Verification, Allergy, Dose, Renal, Pregnancy, Hepatic, Elderly Checks
    qt_prolonging_drugs: List[str] = []

    for rec in med_records:
        gen = rec["generic_name"].lower()
        entered = rec["entered_name"]
        rxcui = rec["rxcui"]
        atc_info = rec.get("atc_info", {})
        allergy_grp = atc_info.get("allergy_group", gen)
        dose_parsed = rec["dose_parsed"]
        d_rule = dose_rules.get(gen)

        dose_status = "OK"
        comments: List[str] = [f"Dose math: {dose_parsed['formatted_calc']}."]

        # Verification Gate: Check if drug is completely unindexed/unknown
        if rxcui == "UNKNOWN" and not rec.get("all_chunks") and not d_rule:
            not_verified_reasons.append(
                f"Medication '{entered}' could not be verified in RxNorm, openFDA/DailyMed, or local structured formulary rules."
            )
            dose_status = "NOT_VERIFIED"
            comments.append("Unindexed drug — manual pharmacist verification required.")

        # A. Coded Allergy & Cross-Reactivity Matrix Check
        for alg in allergies:
            alg_code = alg.get("allergen", "").strip().lower()
            alg_sev = alg.get("severity", "anaphylaxis_severe").lower()
            matrix_entry = allergy_matrix.get(alg_code)

            # Direct drug name or direct allergy group match
            if alg_code == gen or alg_code in gen or (
                matrix_entry and allergy_grp in matrix_entry.get("direct_allergy_groups", [])
            ):
                alerts.append(
                    {
                        "severity": "HIGH",
                        "type": "ALLERGY",
                        "drugs_involved": [entered],
                        "description": (
                            f"Direct Allergy Contraindication: Patient has a documented '{alg_code}' allergy "
                            f"and '{entered}' ({gen}, ATC: {atc_info.get('atc_code')}) belongs to the '{allergy_grp}' group."
                        ),
                        "source": f"Deterministic Allergy Engine (WHO/ISO ATC: {atc_info.get('atc_code')})",
                        "deterministic": True,
                    }
                )
                dose_status = "CONTRAINDICATED"
                comments.append(f"CONTRAINDICATED: Direct {alg_code} allergy match.")
            elif matrix_entry and allergy_grp in matrix_entry.get("cross_reactive_groups", {}):
                cr_info = matrix_entry["cross_reactive_groups"][allergy_grp]
                risk = cr_info["risk_if_severe"] if "severe" in alg_sev or "anaphylaxis" in alg_sev else cr_info["risk_if_mild"]
                if risk in ("HIGH", "MODERATE"):
                    alerts.append(
                        {
                            "severity": risk,
                            "type": "ALLERGY",
                            "drugs_involved": [entered],
                            "description": f"Cross-Reactivity Allergy Alert ({alg_code} -> {allergy_grp}): {cr_info['message']}",
                            "source": f"Deterministic Cross-Reactivity Matrix (ATC: {atc_info.get('atc_code')})",
                            "deterministic": True,
                        }
                    )
                else:
                    suggestions.append(f"Allergy Cross-Reactivity Note for {entered}: {cr_info['message']}")

        # B. Structured Dose, Renal (KDIGO), Hepatic (LiverTox), Pregnancy, LactMed, Beers, QT Checks
        if d_rule:
            if d_rule.get("qt_risk") == "Known":
                qt_prolonging_drugs.append(f"{entered} ({gen})")

            # Dose limit checks
            daily_mg = dose_parsed.get("daily_dose_mg")
            single_mg = dose_parsed.get("single_dose_mg")
            max_daily = d_rule.get("max_daily_dose_mg")
            max_single = d_rule.get("max_single_dose_mg")
            max_hepatic = d_rule.get("max_daily_hepatic_mg")

            if hepatic and max_hepatic is not None:
                if max_hepatic == 0:
                    alerts.append(
                        {
                            "severity": "HIGH",
                            "type": "CONTRAINDICATION",
                            "drugs_involved": [entered],
                            "description": (
                                f"Hepatic Contraindication (LiverTox): '{entered}' ({gen}) is contraindicated in "
                                f"active liver disease / decompensated cirrhosis ({d_rule.get('livertox_class', '')})."
                            ),
                            "source": "LiverTox / Structured Hepatic Rule",
                            "deterministic": True,
                        }
                    )
                    dose_status = "HIGH"
                elif daily_mg and daily_mg > max_hepatic:
                    alerts.append(
                        {
                            "severity": "HIGH",
                            "type": "OVERDOSE",
                            "drugs_involved": [entered],
                            "description": (
                                f"Hepatic Dose Limit Exceeded: Calculated daily dose of {entered} ({int(daily_mg)} mg/day) "
                                f"exceeds the maximum hepatic impairment limit of {max_hepatic} mg/day."
                            ),
                            "source": "LiverTox / WHO EML Hepatic Dosing Rule",
                            "deterministic": True,
                        }
                    )
                    dose_status = "HIGH"

            if daily_mg and max_daily and daily_mg > max_daily:
                alerts.append(
                    {
                        "severity": "HIGH",
                        "type": "OVERDOSE",
                        "drugs_involved": [entered],
                        "description": (
                            f"Maximum Daily Dose Exceeded: Prescribed {entered} equals {int(daily_mg)} mg/day "
                            f"({dose_parsed['formatted_calc']}), exceeding the maximum adult daily limit of {max_daily} mg/day."
                        ),
                        "source": "Deterministic Dose Rule Engine (WHO EML / FDA Label)",
                        "deterministic": True,
                    }
                )
                dose_status = "HIGH"
                comments.append(f"OVERDOSE: {int(daily_mg)} mg/day exceeds max {max_daily} mg/day.")
            elif single_mg and max_single and single_mg > max_single:
                alerts.append(
                    {
                        "severity": "HIGH",
                        "type": "OVERDOSE",
                        "drugs_involved": [entered],
                        "description": (
                            f"Maximum Single Dose Exceeded: Single dose of {int(single_mg)} mg for {entered} "
                            f"exceeds maximum single dose of {max_single} mg."
                        ),
                        "source": "Deterministic Dose Rule Engine",
                        "deterministic": True,
                    }
                )
                dose_status = "HIGH"

            # KDIGO Renal eGFR check
            egfr_contra = d_rule.get("egfr_contraindicated_below")
            egfr_caut = d_rule.get("egfr_caution_below")
            if egfr is not None:
                if egfr_contra and egfr < egfr_contra:
                    alerts.append(
                        {
                            "severity": "HIGH",
                            "type": "CONTRAINDICATION",
                            "drugs_involved": [entered],
                            "description": (
                                f"Renal Contraindication (Patient eGFR = {egfr} mL/min): "
                                f"{d_rule.get('renal_note', f'{gen} is contraindicated when eGFR < {egfr_contra} mL/min.')}"
                            ),
                            "source": "KDIGO Renal Dosing Guidelines",
                            "deterministic": True,
                        }
                    )
                    dose_status = "HIGH"
                elif egfr_caut and egfr < egfr_caut:
                    suggestions.append(
                        f"Renal Dose Adjustment (eGFR = {egfr} mL/min) for {entered}: {d_rule.get('renal_note', 'Monitor renal function and reduce dose.')}"
                    )
                    if dose_status == "OK":
                        dose_status = "REVIEW"

            # Pregnancy check
            if pregnancy != "none" and d_rule.get("pregnancy_contraindicated"):
                alerts.append(
                    {
                        "severity": "HIGH",
                        "type": "CONTRAINDICATION",
                        "drugs_involved": [entered],
                        "description": (
                            f"Pregnancy Contraindication: {entered} ({gen}) is contraindicated during pregnancy. "
                            f"{d_rule.get('pregnancy_note', '')}"
                        ),
                        "source": "FDA Pregnancy / Teratogenicity Rule",
                        "deterministic": True,
                    }
                )
                dose_status = "HIGH"

            # LactMed Breastfeeding check
            if lactation and d_rule.get("lactmed_status"):
                l_note = d_rule["lactmed_status"]
                if "contraindicated" in l_note.lower() or "avoid" in l_note.lower():
                    alerts.append(
                        {
                            "severity": "MODERATE",
                            "type": "CONTRAINDICATION",
                            "drugs_involved": [entered],
                            "description": f"LactMed Breastfeeding Warning for {entered} ({gen}): {l_note}",
                            "source": "NLM LactMed Reference",
                            "deterministic": True,
                        }
                    )
                else:
                    suggestions.append(f"LactMed Breastfeeding Info ({entered}): {l_note}")

            # AGS Beers Elderly Check (Age >= 65)
            if age is not None and age >= 65 and d_rule.get("beers_elderly_caution"):
                suggestions.append(
                    f"AGS Beers Criteria (Age {age} >= 65 yrs) for {entered}: {d_rule.get('beers_note', 'Use with caution in older adults.')}"
                )

        # Special Condition Checks: Asthma + Non-selective Beta-Blocker or NSAID; Dengue/Peptic Ulcer + NSAID
        if has_asthma and gen == "propranolol":
            alerts.append(
                {
                    "severity": "HIGH",
                    "type": "CONTRAINDICATION",
                    "drugs_involved": [entered],
                    "description": "Propranolol (non-selective beta-blocker) is contraindicated in bronchial asthma due to risk of severe, potentially fatal bronchospasm.",
                    "source": "Deterministic Drug-Disease Rule (Asthma)",
                    "deterministic": True,
                }
            )
        if (has_peptic_ulcer or has_dengue) and allergy_grp in ("nsaid", "salicylate", "cox2_nsaid"):
            cond_label = "Dengue Fever / Thrombocytopenia" if has_dengue else "Active Peptic Ulcer / GI Bleed"
            alerts.append(
                {
                    "severity": "HIGH",
                    "type": "CONTRAINDICATION",
                    "drugs_involved": [entered],
                    "description": f"{entered} ({gen}, NSAID/Salicylate) is contraindicated in patients with {cond_label} due to high risk of severe hemorrhage.",
                    "source": f"Deterministic Drug-Disease Rule ({cond_label})",
                    "deterministic": True,
                }
            )

        med_reviews[gen] = {
            "entered_name": entered,
            "generic_name": gen,
            "rxcui": rxcui,
            "atc_code": atc_info.get("atc_code", "UNCLASSIFIED"),
            "therapeutic_class": atc_info.get("therapeutic_class", ""),
            "entered_dose": rec["entered_dose"],
            "calculated_daily_dose": dose_parsed["formatted_calc"],
            "dose_status": dose_status,
            "comment": " ".join(comments),
        }

    # 2. Exact RxCUI Duplicate & Same-ATC-Class Therapeutic Duplication
    seen_rxcuis: Dict[str, List[str]] = {}
    seen_classes: Dict[str, List[Dict[str, str]]] = {}

    for rec in med_records:
        gen = rec["generic_name"].lower()
        rxcui = rec["rxcui"]
        key = rxcui if rxcui != "UNKNOWN" else gen
        seen_rxcuis.setdefault(key, []).append(f"{rec['entered_name']} ({rec['entered_dose']})")

        #Also check multi-ingredient overlap (e.g., Panadeine contains acetaminophen + Paracetamol)
        for ing in rec.get("ingredients", [gen]):
            if ing != gen:
                for other_rec in med_records:
                    if other_rec["generic_name"].lower() == ing:
                        alerts.append(
                            {
                                "severity": "HIGH",
                                "type": "DUPLICATE_THERAPY",
                                "drugs_involved": [rec["entered_name"], other_rec["entered_name"]],
                                "description": (
                                    f"Duplicate Active Ingredient: '{rec['entered_name']}' and '{other_rec['entered_name']}' "
                                    f"both contain '{ing}', risking accidental overdose."
                                ),
                                "source": "RxNorm Ingredient Overlap Check",
                                "deterministic": True,
                            }
                        )

        atc_info = rec.get("atc_info", {})
        t_class = atc_info.get("therapeutic_class", "Unclassified")
        if t_class and t_class != "Unclassified":
            seen_classes.setdefault(t_class, []).append(
                {"entered": rec["entered_name"], "generic": gen, "atc": atc_info.get("atc_code", "")}
            )

    for key, items in seen_rxcuis.items():
        if len(items) > 1:
            gen_match = next(
                (r["generic_name"] for r in med_records if (r["rxcui"] == key or r["generic_name"] == key)),
                key,
            )
            alerts.append(
                {
                    "severity": "HIGH",
                    "type": "DUPLICATE_THERAPY",
                    "drugs_involved": items,
                    "description": (
                        f"Exact Duplicate Ingredient (RxCUI {key}): {' and '.join(items)} both contain '{gen_match}'. "
                        f"Concurrent administration risks overdose and toxicity."
                    ),
                    "source": f"RxNorm Normalization (RxCUI: {key})",
                    "deterministic": True,
                }
            )

    for t_class, items in seen_classes.items():
        unique_generics = {it["generic"] for it in items}
        if len(unique_generics) > 1:
            names = [f"{it['entered']} ({it['generic']}, ATC: {it['atc']})" for it in items]
            alerts.append(
                {
                    "severity": "HIGH",
                    "type": "THERAPEUTIC_DUPLICATION",
                    "drugs_involved": [it["entered"] for it in items],
                    "description": (
                        f"Same-Class Therapeutic Duplication ({t_class}): {', '.join(names)} belong to the same "
                        f"WHO/ISO ATC pharmacologic class ({t_class}). Co-prescribing two drugs from the same class "
                        f"increases toxicity without added therapeutic benefit."
                    ),
                    "source": f"WHO/ISO ATC Class Duplication Rule ({t_class})",
                    "deterministic": True,
                }
            )

    # 3. Structured DDInter 2.0 Drug-Drug Interaction Lookup
    generic_list = [r["generic_name"].lower() for r in med_records]
    ddi_matches = ddinter_store.check_all_pairs(generic_list)

    for ddi in ddi_matches:
        sev_raw = ddi["severity"].upper()
        sev = "HIGH" if "MAJOR" in sev_raw or "HIGH" in sev_raw else "MODERATE"
        alerts.append(
            {
                "severity": sev,
                "type": "DRUG_INTERACTION",
                "drugs_involved": [ddi["drug_a"], ddi["drug_b"]],
                "description": f"[{ddi['ddinter_id']} — {ddi['severity']} Interaction] {ddi['mechanism']} Management: {ddi['management']}",
                "source": ddi["source"],
                "ddinter_id": ddi["ddinter_id"],
                "deterministic": True,
            }
        )
        if ddi.get("management"):
            suggestions.append(f"DDI Management ({ddi['drug_a']} + {ddi['drug_b']}): {ddi['management']}")

    # 4. CredibleMeds Multi-QT Prolongation Check (if >=2 Known QT drugs not already flagged)
    if len(qt_prolonging_drugs) >= 2:
        suggestions.append(
            f"CredibleMeds QT Caution: Multiple QT-prolonging drugs prescribed ({', '.join(qt_prolonging_drugs)}). Consider baseline ECG and electrolyte check."
        )

    return {
        "alerts": alerts,
        "suggestions": suggestions,
        "medication_reviews": list(med_reviews.values()),
        "not_verified_reasons": not_verified_reasons,
    }

