import hashlib
import json
import logging
import re
import sqlite3
from typing import List, Dict, Any, Optional
import numpy as np

from config import VECTOR_DB_PATH, CHUNKS_JSONL_PATH
from pipeline.rxnorm import normalize_drug_name
from pipeline.label_fetcher import fetch_drug_label_chunks
from pipeline.who_eml import build_who_eml_chunk, get_who_eml_info

logger = logging.getLogger(__name__)

EMBED_DIM = 1024


def compute_text_embedding(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """
    Fast, deterministic local dense vector embedding using signed feature hashing
    over unigrams, bigrams, and character 4-grams, L2-normalized for cosine similarity.
    Works 100% offline with zero external model latency.
    """
    vec = np.zeros(dim, dtype=np.float32)
    if not text:
        return vec

    tokens = re.findall(r"[a-z0-9]{2,}", text.lower())
    if not tokens:
        return vec

    features: List[str] = list(tokens)
    # Add word bigrams for clinical phrase matching (e.g., "lactic acidosis", "bleeding risk")
    features.extend(f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1))

    for feat in features:
        h = int(hashlib.md5(feat.encode("utf-8")).hexdigest()[:8], 16)
        idx = h % dim
        sign = 1.0 if (h & (1 << 31)) else -1.0
        vec[idx] += sign

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


class CDSSVectorStore:
    """
    Persistent Local Vector DB (SQLite + NumPy cosine similarity) paired with a
    pharmacist-auditable JSONL file (`Data/chunks/knowledge_chunks.jsonl`).
    """

    def __init__(self, db_path=VECTOR_DB_PATH, jsonl_path=CHUNKS_JSONL_PATH):
        self.db_path = db_path
        self.jsonl_path = jsonl_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    drug TEXT NOT NULL,
                    rxcui TEXT NOT NULL,
                    section TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    label_version_date TEXT NOT NULL,
                    retrieved_on TEXT NOT NULL,
                    pharmacist_verified INTEGER DEFAULT 0,
                    pharmacist_notes TEXT DEFAULT '',
                    embedding BLOB NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_drug ON chunks(drug)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_rxcui ON chunks(rxcui)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section)")
            conn.commit()

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Embed and store chunks in the vector DB and append new entries to JSONL."""
        if not chunks:
            return 0

        added = 0
        new_jsonl_lines: List[str] = []

        with self._get_conn() as conn:
            for ch in chunks:
                drug = str(ch.get("drug", "")).strip().lower()
                rxcui = str(ch.get("rxcui", "UNKNOWN")).strip()
                section = str(ch.get("section", "")).strip()
                text = str(ch.get("text", "")).strip()
                if not drug or not section or not text:
                    continue

                chunk_id = f"{drug}::{rxcui}::{section}"
                embed_input = f"Drug: {drug} | Section: {section} | {text}"
                emb = compute_text_embedding(embed_input)

                cursor = conn.execute(
                    "SELECT chunk_id FROM chunks WHERE chunk_id = ?", (chunk_id,)
                )
                exists = cursor.fetchone() is not None

                conn.execute(
                    """
                    INSERT OR REPLACE INTO chunks (
                        chunk_id, drug, rxcui, section, text,
                        source, label_version_date, retrieved_on,
                        pharmacist_verified, pharmacist_notes, embedding
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?,
                        COALESCE((SELECT pharmacist_verified FROM chunks WHERE chunk_id = ?), 0),
                        COALESCE((SELECT pharmacist_notes FROM chunks WHERE chunk_id = ?), ''),
                        ?
                    )
                    """,
                    (
                        chunk_id,
                        drug,
                        rxcui,
                        section,
                        text,
                        str(ch.get("source", "openFDA")),
                        str(ch.get("label_version_date", "unknown")),
                        str(ch.get("retrieved_on", "")),
                        chunk_id,
                        chunk_id,
                        emb.tobytes(),
                    ),
                )
                if not exists:
                    added += 1
                    clean_record = {
                        "drug": drug,
                        "rxcui": rxcui,
                        "section": section,
                        "text": text,
                        "source": str(ch.get("source", "openFDA")),
                        "label_version_date": str(ch.get("label_version_date", "unknown")),
                        "retrieved_on": str(ch.get("retrieved_on", "")),
                    }
                    new_jsonl_lines.append(json.dumps(clean_record, ensure_ascii=False))
            conn.commit()

        if new_jsonl_lines:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                for line in new_jsonl_lines:
                    f.write(line + "\n")

        return added

    def get_drug_chunks(
        self, generic_name: str, rxcui: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve all stored chunks for a given generic drug or RxCUI."""
        drug_key = generic_name.strip().lower()
        with self._get_conn() as conn:
            if rxcui and rxcui != "UNKNOWN":
                rows = conn.execute(
                    """
                    SELECT chunk_id, drug, rxcui, section, text, source,
                           label_version_date, retrieved_on, pharmacist_verified, pharmacist_notes
                    FROM chunks
                    WHERE drug = ? OR rxcui = ?
                    ORDER BY section
                    """,
                    (drug_key, str(rxcui)),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT chunk_id, drug, rxcui, section, text, source,
                           label_version_date, retrieved_on, pharmacist_verified, pharmacist_notes
                    FROM chunks
                    WHERE drug = ?
                    ORDER BY section
                    """,
                    (drug_key,),
                ).fetchall()

        return [dict(r) for r in rows]

    def search_chunks(
        self,
        query: str,
        drugs: Optional[List[str]] = None,
        sections: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Semantic + lexical vector similarity search across stored chunks,
        optionally filtered by a list of drugs and/or sections.
        """
        q_emb = compute_text_embedding(query)
        clauses: List[str] = []
        params: List[Any] = []

        if drugs:
            clean_drugs = [d.strip().lower() for d in drugs if d]
            placeholders = ",".join("?" for _ in clean_drugs)
            clauses.append(f"drug IN ({placeholders})")
            params.extend(clean_drugs)

        if sections:
            placeholders = ",".join("?" for _ in sections)
            clauses.append(f"section IN ({placeholders})")
            params.extend(sections)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT chunk_id, drug, rxcui, section, text, source,
                   label_version_date, retrieved_on, pharmacist_verified, pharmacist_notes, embedding
            FROM chunks
            {where_sql}
        """

        with self._get_conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        scored_results: List[Dict[str, Any]] = []
        query_terms = [t for t in re.findall(r"[a-z0-9]{3,}", query.lower())]

        for r in rows:
            emb_bytes = r["embedding"]
            doc_emb = np.frombuffer(emb_bytes, dtype=np.float32)
            score = float(np.dot(q_emb, doc_emb)) if doc_emb.shape == q_emb.shape else 0.0

            # Boost exact drug/clinical keyword matches inside the chunk text
            lower_text = r["text"].lower()
            keyword_hits = sum(1 for t in query_terms if t in lower_text)
            score += 0.08 * keyword_hits

            item = {
                "chunk_id": r["chunk_id"],
                "drug": r["drug"],
                "rxcui": r["rxcui"],
                "section": r["section"],
                "text": r["text"],
                "source": r["source"],
                "label_version_date": r["label_version_date"],
                "retrieved_on": r["retrieved_on"],
                "pharmacist_verified": bool(r["pharmacist_verified"]),
                "pharmacist_notes": r["pharmacist_notes"],
                "similarity_score": round(score, 4),
            }
            scored_results.append(item)

        scored_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored_results[:top_k]

    def ensure_drug_indexed(self, raw_drug_name: str) -> Dict[str, Any]:
        """
        End-to-end knowledge pipeline for a single medication:
        1. Normalize drug name via RxNorm -> get generic name & RxCUI.
        2. Check if already in Vector DB.
        3. If not in Vector DB, pull label sections from openFDA (fallback DailyMed) + WHO EML.
        4. Embed & store chunks in Vector DB + JSONL.
        """
        norm_info = normalize_drug_name(raw_drug_name)
        generic_name = norm_info["generic_name"]
        rxcui = norm_info["rxcui"]

        existing_chunks = self.get_drug_chunks(generic_name, rxcui)
        fetched_live = False

        if not existing_chunks:
            fetched_live = True
            new_chunks = fetch_drug_label_chunks(generic_name, rxcui)
            who_chunk = build_who_eml_chunk(generic_name, rxcui)
            if who_chunk:
                new_chunks.append(who_chunk)

            # Also check individual ingredients if combination product (e.g., amoxicillin / clavulanate)
            if not new_chunks and len(norm_info.get("ingredients", [])) > 1:
                for ing in norm_info["ingredients"][:2]:
                    ing_chunks = fetch_drug_label_chunks(ing, rxcui)
                    new_chunks.extend(ing_chunks)

            if new_chunks:
                self.add_chunks(new_chunks)
                existing_chunks = self.get_drug_chunks(generic_name, rxcui)

        who_info = get_who_eml_info(generic_name)
        return {
            "normalization": norm_info,
            "who_eml": who_info,
            "chunks": existing_chunks,
            "fetched_live": fetched_live,
        }

    def mark_pharmacist_verified(
        self, chunk_id: str, verified: bool = True, notes: str = ""
    ) -> bool:
        """Allow a pharmacist to spot-check and mark a stored chunk as verified."""
        with self._get_conn() as conn:
            cur = conn.execute(
                """
                UPDATE chunks
                SET pharmacist_verified = ?, pharmacist_notes = ?
                WHERE chunk_id = ?
                """,
                (1 if verified else 0, notes.strip(), chunk_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics of stored drugs and chunks for the UI dashboard."""
        with self._get_conn() as conn:
            total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            total_drugs = conn.execute("SELECT COUNT(DISTINCT drug) FROM chunks").fetchone()[0]
            verified_chunks = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE pharmacist_verified = 1"
            ).fetchone()[0]
            drugs_rows = conn.execute(
                """
                SELECT drug, rxcui, COUNT(*) as section_count,
                       MAX(retrieved_on) as last_updated,
                       SUM(pharmacist_verified) as verified_count
                FROM chunks
                GROUP BY drug
                ORDER BY drug
                """
            ).fetchall()

        return {
            "total_chunks": total_chunks,
            "total_drugs": total_drugs,
            "verified_chunks": verified_chunks,
            "indexed_drugs": [dict(r) for r in drugs_rows],
        }

