import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta

import numpy as np
import faiss
from dotenv import load_dotenv
import os

load_dotenv()

# OpenAI client (new SDK preferred, fallback to legacy)
try:
    from openai import OpenAI
    _client = OpenAI()
    _use_client = True
except Exception:
    _client = None
    _use_client = False
    import openai  # type: ignore
    openai.api_key = os.getenv("OPENAI_API_KEY")

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

INDEX_PATH = Path("embeddings/faiss.index")
META_PATH = Path("embeddings/metadata.pkl")

# ─────────────────────────────────────────────────────────────
# Embedding (query)
# ─────────────────────────────────────────────────────────────
def _embed_query_client(text: str) -> np.ndarray:
    resp = _client.embeddings.create(model=EMBED_MODEL, input=text)  # type: ignore
    vec = resp.data[0].embedding
    return np.asarray(vec, dtype=np.float32)

def _embed_query_legacy(text: str) -> np.ndarray:
    resp = openai.Embedding.create(model=EMBED_MODEL, input=text)  # type: ignore
    vec = resp["data"][0]["embedding"]
    return np.asarray(vec, dtype=np.float32)

def embed_query(text: str) -> np.ndarray:
    arr = _embed_query_client(text) if _use_client else _embed_query_legacy(text)
    if arr.shape != (EMBED_DIM,):
        raise ValueError(f"Unexpected embedding shape {arr.shape}")
    return arr

# ─────────────────────────────────────────────────────────────
# Load FAISS + metadata
# ─────────────────────────────────────────────────────────────
def load_resources():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Missing FAISS index or metadata. Run embed_and_store.py first.")
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# ─────────────────────────────────────────────────────────────
# Base semantic search
# ─────────────────────────────────────────────────────────────
def search(query: str, k: int = 5) -> List[Tuple[int, float, Dict]]:
    index, metadata = load_resources()
    qvec = embed_query(query).reshape(1, -1)
    D, I = index.search(qvec, max(k, 5))
    results: List[Tuple[int, float, Dict]] = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        meta = metadata.get(int(idx), {})
        results.append((int(idx), float(dist), meta))
    return results

# ─────────────────────────────────────────────────────────────
# Meetings & Date-window utilities
# ─────────────────────────────────────────────────────────────
def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except Exception:
        return None

def rerank(results: List[Tuple[int, float, Dict]], prefer_meetings: bool = False, prefer_recent: bool = False):
    """
    Re-rank by:
      1) folder == 'Meetings' boost (if prefer_meetings)
      2) newer meeting_date (if prefer_recent)
      3) FAISS distance (smaller is better)
    """
    def score(item):
        _, dist, meta = item
        base = -dist  # smaller is better → invert
        folder_bonus = 1000 if (prefer_meetings and str(meta.get("folder", "")).lower() == "meetings") else 0
        d = _parse_iso(meta.get("meeting_date"))
        date_bonus = d.toordinal() if (prefer_recent and d) else 0
        return folder_bonus * 1_000_000 + date_bonus * 10 + base
    return sorted(results, key=score, reverse=True)

def search_meetings(query: str, k: int = 5, prefer_recent: bool = True) -> List[Tuple[int, float, Dict]]:
    raw = search(query, k=max(k, 50))  # widen pool
    reranked = rerank(raw, prefer_meetings=True, prefer_recent=prefer_recent)
    return reranked[:k]

def filter_by_date_range(results: List[Tuple[int, float, Dict]], start: datetime, end: datetime) -> List[Tuple[int, float, Dict]]:
    kept: List[Tuple[int, float, Dict]] = []
    for rid, dist, meta in results:
        d = _parse_iso(meta.get("meeting_date"))
        if d and start <= d <= end:
            kept.append((rid, dist, meta))
    return kept

def rerank_for_recency(results: List[Tuple[int, float, Dict]], favor_recent: bool = True) -> List[Tuple[int, float, Dict]]:
    def key(x):
        _, dist, meta = x
        base = -dist
        d = _parse_iso(meta.get("meeting_date"))
        rec = d.toordinal() if (favor_recent and d) else 0
        return (rec * 10) + base
    return sorted(results, key=key, reverse=True)

def search_in_date_window(query: str, start: datetime, end: datetime, k: int = 5) -> List[Tuple[int, float, Dict]]:
    """
    Pull a broader pool, filter to [start, end] by stored meeting_date, then rerank.
    """
    pool = search(query, k=max(k, 200))
    windowed = filter_by_date_range(pool, start, end)
    if not windowed:
        return []
    return rerank_for_recency(windowed)[:k]

if __name__ == "__main__":
    # Example quick test
    hits = search_meetings("decisions on roadmap", k=5)
    for i, (vid, dist, meta) in enumerate(hits, 1):
        print(f"{i}. ID={vid}  dist={dist:.4f}  file={meta.get('filename')}  chunk={meta.get('chunk_id')}  date={meta.get('meeting_date')}  folder={meta.get('folder')}")
        print(meta.get("text_preview", "")[:200], "\n---")
