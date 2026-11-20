"""
DDInter 2.0 Structured Drug-Drug Interaction Database Loader & Query Engine.
Replaces narrative FDA label guessing with deterministic SQLite lookups.
Supports both the built-in core hospital formulary matrix (`Data/ddinter_core_seed.json`)
and official DDInter 2.0 downloadable CSV files (`ddinter_downloads_code_*.csv`).
"""

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import DDINTER_DB_PATH, DDINTER_SEED_PATH
from engine.trace_logger import iso_logger


class DDInterStore:
    """SQLite-backed structured Drug-Drug Interaction store modeled on DDInter 2.0."""

    def __init__(self, db_path: Path = DDINTER_DB_PATH, seed_path: Path = DDINTER_SEED_PATH) -> None:
        self.db_path = db_path
        self.seed_path = seed_path
        self._init_db()
        self._ensure_seeded()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ddi_records (
                    pair_key TEXT PRIMARY KEY,
                    ddinter_id TEXT NOT NULL,
                    drug_a TEXT NOT NULL,
                    drug_b TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    mechanism TEXT NOT NULL,
                    management TEXT NOT NULL,
                    source TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ddi_drugs ON ddi_records(drug_a, drug_b)")
            conn.commit()

    @staticmethod
    def make_pair_key(drug_1: str, drug_2: str) -> str:
        """Create canonical alphabetical pair key so (A, B) and (B, A) map identically."""
        d1, d2 = sorted([drug_1.strip().lower(), drug_2.strip().lower()])
        return f"{d1}::{d2}"

    def _ensure_seeded(self) -> None:
        with self._get_conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM ddi_records").fetchone()[0]
        if count == 0 and self.seed_path.exists():
            self.load_seed_json(self.seed_path)

    def load_seed_json(self, json_path: Path = DDINTER_SEED_PATH) -> int:
        """Load structured interaction records from JSON seed file."""
        if not json_path.exists():
            return 0
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        interactions = data.get("interactions", [])
        inserted = 0
        with self._get_conn() as conn:
            for item in interactions:
                da = str(item.get("drug_a", "")).strip().lower()
                db = str(item.get("drug_b", "")).strip().lower()
                if not da or not db or da == db:
                    continue
                d1, d2 = sorted([da, db])
                pair_key = f"{d1}::{d2}"
                conn.execute(
                    """
                    INSERT OR REPLACE INTO ddi_records (
                        pair_key, ddinter_id, drug_a, drug_b, severity, mechanism, management, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pair_key,
                        str(item.get("ddinter_id", "DDInter-CORE")),
                        d1,
                        d2,
                        str(item.get("severity", "Moderate")),
                        str(item.get("mechanism", "")),
                        str(item.get("management", "")),
                        "DDInter 2.0 Structured DB",
                    ),
                )
                inserted += 1
            conn.commit()
        iso_logger.info(f"Loaded {inserted} structured DDInter 2.0 interaction records into {self.db_path.name}")
        return inserted

    def import_ddinter_csv(self, csv_path: str | Path) -> int:
        """
        Import an official DDInter 2.0 downloadable CSV file (`ddinter_downloads_code_*.csv`).
        Expected columns: DDInterID_A, Drug_A, DDInterID_B, Drug_B, Level
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"DDInter CSV not found: {path}")

        inserted = 0
        with open(path, "r", encoding="utf-8", errors="replace") as f, self._get_conn() as conn:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                da = (row.get("Drug_A") or row.get("drug_a") or "").strip().lower()
                db = (row.get("Drug_B") or row.get("drug_b") or "").strip().lower()
                level = (row.get("Level") or row.get("severity") or "Moderate").strip()
                if not da or not db or da == db or level.lower() == "unknown":
                    continue
                d1, d2 = sorted([da, db])
                pair_key = f"{d1}::{d2}"
                dd_id = f"{row.get('DDInterID_A', 'DDI')}-{row.get('DDInterID_B', idx)}"
                conn.execute(
                    """
                    INSERT OR IGNORE INTO ddi_records (
                        pair_key, ddinter_id, drug_a, drug_b, severity, mechanism, management, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pair_key,
                        dd_id,
                        d1,
                        d2,
                        level,
                        str(row.get("Mechanism", f"Documented {level} interaction in DDInter 2.0 between {d1} and {d2}.")),
                        str(row.get("Management", "Review clinical necessity, monitor closely, or consider alternative therapy.")),
                        f"DDInter 2.0 CSV ({path.name})",
                    ),
                )
                inserted += 1
            conn.commit()
        return inserted

    def lookup_pair(self, drug_1: str, drug_2: str) -> Optional[Dict[str, Any]]:
        """Deterministic O(1) lookup for a drug-drug interaction between two normalized generic names."""
        if not drug_1 or not drug_2:
            return None
        d1, d2 = sorted([drug_1.strip().lower(), drug_2.strip().lower()])
        if d1 == d2:
            return None
        pair_key = f"{d1}::{d2}"
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM ddi_records WHERE pair_key = ?", (pair_key,)
            ).fetchone()
        return dict(row) if row else None

    def check_all_pairs(self, generic_drugs: List[str]) -> List[Dict[str, Any]]:
        """Check all unique pairs in a prescription list against DDInter 2.0."""
        matches: List[Dict[str, Any]] = []
        seen_keys = set()
        for i in range(len(generic_drugs)):
            for j in range(i + 1, len(generic_drugs)):
                da, db = generic_drugs[i], generic_drugs[j]
                pkey = self.make_pair_key(da, db)
                if pkey in seen_keys:
                    continue
                seen_keys.add(pkey)
                rec = self.lookup_pair(da, db)
                if rec:
                    matches.append(rec)
        return matches

    def get_total_records(self) -> int:
        with self._get_conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM ddi_records").fetchone()[0]

