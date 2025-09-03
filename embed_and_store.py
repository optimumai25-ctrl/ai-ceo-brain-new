import os
import time
import pickle
import re
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import faiss
from dotenv import load_dotenv
from tqdm import tqdm

from chunk_utils import simple_chunks

# ─────────────────────────────────────────────────────────────
# OpenAI (legacy-compatible; uses Streamlit secrets if present)
# ─────────────────────────────────────────────────────────────
load_dotenv()

# Prefer new SDK if available
try:
    from openai import OpenAI
    _client = OpenAI()
    _use_client = True
except Exception:
    _client = None
    _use_client = False
    import openai  # type: ignore
    # If running in Streamlit, pick key from secrets; else ENV
    try:
        import streamlit as st  # type: ignore
        openai.api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        openai.api_key = os.getenv("OPENAI_API_KEY")

# ─────────────────────────────────────────────────────────────
# Paths & Config
# ─────────────────────────────────────────────────────────────
PARSED_DIR = Path("parsed_data")
EMBED_DIR = Path("embeddings")
EMBED_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
INDEX_PATH = EMBED_DIR / "faiss.index"
META_PATH = EMBED_DIR / "metadata.pkl"
REPORT_CSV = EMBED_DIR / "embedding_report.csv"

# FAISS index (ID-mapped)
_base_index = faiss.IndexFlatL2(EMBED_DIM)
_index = faiss.IndexIDMap2(_base_index)

# runtime metadata map: vector_id -> dict
_metadata: Dict[int, Dict] = {}
_next_id = 0

# ─────────────────────────────────────────────────────────────
# Helpers: filename date & headers
# ─────────────────────────────────────────────────────────────
# Canonical: 2025-09-02_Meeting-Summary.docx (or .pdf/.xlsx)
_CANON = re.compile(
    r'^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})_Meeting-Summary\b',
    re.IGNORECASE
)

def date_from_filename(fname: str) -> Optional[str]:
    """
    Extracts ISO date 'YYYY-MM-DD' from a canonical daily-summary filename.
    Returns None if not matched.
    """
    base = Path(fname).stem
    m = _CANON.match(base)
    if not m:
        return None
    y, mo, d = int(m.group('y')), int(m.group('m')), int(m.group('d'))
    try:
        return datetime(y, mo, d).strftime("%Y-%m-%d")
    except Exception:
        return None

def extract_folder_and_file_headers(text: str) -> (str, str):
    """
    parsed_data/*.txt files start with:
      [FOLDER]: <subfolder>
      [FILE]: <original_name.ext>
    """
    folder, original_file = "", ""
    lines = text.splitlines()[:5]
    for ln in lines:
        if ln.startswith("[FOLDER]:"):
            folder = ln.split(":", 1)[1].strip()
        elif ln.startswith("[FILE]:"):
            original_file = ln.split(":", 1)[1].strip()
    return folder, original_file

# ─────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────
def _embed_legacy(text: str) -> np.ndarray:
    resp = openai.Embedding.create(model=EMBED_MODEL, input=text)  # type: ignore
    vec = resp["data"][0]["embedding"]
    return np.asarray(vec, dtype=np.float32)

def _embed_client(text: str) -> np.ndarray:
    resp = _client.embeddings.create(model=EMBED_MODEL, input=text)  # type: ignore
    vec = resp.data[0].embedding
    return np.asarray(vec, dtype=np.float32)

def get_embedding(text: str) -> Optional[np.ndarray]:
    """
    Robust embedding with retries & shape check.
    """
    for attempt in range(4):
        try:
            arr = _embed_client(text) if _use_client else _embed_legacy(text)
            if arr.shape != (EMBED_DIM,):
                raise ValueError(f"Unexpected embedding shape {arr.shape}")
            return arr
        except Exception as e:
            wait = 1.5 ** attempt
            print(f"Embedding error (attempt {attempt + 1}): {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    print("Failed to embed after retries.")
    return None

def add_to_index(vec: np.ndarray, vid: int) -> None:
    _index.add_with_ids(vec.reshape(1, -1), np.array([vid], dtype=np.int64))

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    global _next_id

    if not PARSED_DIR.exists():
        print(f"Missing folder: {PARSED_DIR.resolve()}")
        return

    files = sorted([p for p in PARSED_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])
    if not files:
        print("No .txt files found in parsed_data.")
        return

    print(f"Found {len(files)} files to embed (chunking enabled).")
    report_rows: List[tuple] = [("filename", "folder", "meeting_date", "chunks", "chars")]

    for fp in tqdm(files, desc="Embedding"):
        text = fp.read_text(encoding="utf-8").strip()
        if not text:
            print(f"Skipping empty: {fp.name}")
            continue

        # Pull headers & date
        folder_label, original_parsed_name = extract_folder_and_file_headers(text)
        # Prefer original file header; fall back to txt name
        raw_name_for_date = original_parsed_name or fp.name
        meeting_date_iso = date_from_filename(raw_name_for_date)

        # Chunking
        chunks = simple_chunks(text, max_chars=3500, overlap=300)
        if not chunks:
            chunks = [{"chunk_id": 0, "text": text[:3500]}]

        total_chars = sum(len(ch["text"]) for ch in chunks)
        report_rows.append((fp.name, folder_label or "", meeting_date_iso or "", len(chunks), total_chars))

        # Embed + index
        for ch in chunks:
            vec = get_embedding(ch["text"])
            if vec is None:
                print(f"Skipping chunk {ch['chunk_id']} of {fp.name} due to embedding failure.")
                continue

            add_to_index(vec, _next_id)
            _metadata[_next_id] = {
                "filename": fp.name,
                "path": str(fp),
                "chunk_id": ch["chunk_id"],
                "text_preview": ch["text"][:1000],
                "folder": folder_label,              # NEW
                "meeting_date": meeting_date_iso      # NEW (YYYY-MM-DD or None)
            }
            _next_id += 1

    # Persist FAISS + metadata
    faiss.write_index(_index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(_metadata, f)

    # Health report
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(report_rows)

    print(f"✅ Saved FAISS index to {INDEX_PATH}")
    print(f"✅ Saved metadata for {len(_metadata)} vectors to {META_PATH}")
    print(f"📝 Wrote embedding health report to {REPORT_CSV}")

if __name__ == "__main__":
    main()
