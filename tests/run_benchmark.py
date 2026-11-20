"""
Pharmacist-Reviewed Clinical Accuracy Benchmark Runner (`tests/run_benchmark.py`).
Evaluates the CDSS engine against `tests/clinical_test_cases.json` and measures:
- True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)
- Exact 4-State Accuracy (`ALL_RIGHT`, `SUGGESTION`, `ALERT`, `NOT_VERIFIED`)
- False Positive Rate (FPR) and False Negative Rate (FNR)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.cdss_evaluator import CDSSEvaluator


def run_benchmark(use_llm: bool = False) -> int:
    cases_path = Path(__file__).resolve().parent / "clinical_test_cases.json"
    with open(cases_path, "r", encoding="utf-8") as f:
        suite = json.load(f)

    cases = suite.get("cases", [])
    evaluator = CDSSEvaluator()

    passed = 0
    tp = fp = tn = fn = 0

    print("=" * 88)
    print(f"CDSS CLINICAL BENCHMARK ({len(cases)} Cases | Deterministic Rules + DDInter 2.0 + WHO/ISO ATC)")
    print("=" * 88)

    for c in cases:
        cid = c["case_id"]
        expected = c["expected_status"]
        res = evaluator.evaluate_prescription(
            patient_id=f"BENCH-{cid}",
            patient_name="Benchmark Patient",
            medications=c["medications"],
            patient_notes=c.get("patient_notes", ""),
            coded_allergies=c.get("coded_allergies"),
            patient_age=c.get("patient_age"),
            egfr=c.get("egfr"),
            pregnancy_status=c.get("pregnancy_status", "none"),
            use_llm_explainer=use_llm,
        )
        actual = res["overall_status"]
        ok = actual == expected
        if ok:
            passed += 1

        # Binary Alert vs Safe/Unverified tracking for FPR / FNR
        if expected == "ALERT" and actual == "ALERT":
            tp += 1
        elif expected == "ALL_RIGHT" and actual == "ALERT":
            fp += 1
        elif expected == "ALL_RIGHT" and actual == "ALL_RIGHT":
            tn += 1
        elif expected == "ALERT" and actual != "ALERT":
            fn += 1

        mark = "PASS" if ok else "FAIL"
        lat = res.get("system_trace", {}).get("latency_ms", {}).get("total_latency_ms", 0)
        print(f"[{mark}] {cid}: Expected={expected:<12} | Actual={actual:<12} | {lat:>6.1f}ms | {c['description']}")
        if not ok:
            print(f"       -> Summary: {res.get('summary')}")
            print(f"       -> Alerts:  {res.get('alerts')}")

    total = len(cases)
    fpr = (fp / (fp + tn) * 100.0) if (fp + tn) else 0.0
    fnr = (fn / (fn + tp) * 100.0) if (fn + tp) else 0.0

    print("-" * 88)
    print(f"Exact 4-State Match: {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"Alert Confusion Matrix: TP={tp}, TN={tn}, FP={fp} (FPR={fpr:.1f}%), FN={fn} (FNR={fnr:.1f}%)")
    print("=" * 88)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(run_benchmark(use_llm=False))

