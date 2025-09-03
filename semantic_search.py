import pickle
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import faiss
from dotenv import load_dotenv
import os

# Load environment
load_dotenv()

# OpenAI client (new SDK style)
try:
    from openai import OpenAI
    client = OpenAI()
    use_client = True
except Exception:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    use_client = False

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

INDEX_PATH = Path("embeddings/faiss.index")
META_PATH = Path("embeddings/metadata.pkl")

def embed_query(text: str) -> np.ndarray:
    if use_client:
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=text
        )
        vec = response.data[0].embedding
    else:
        response = openai.Embedding.create(
            model=EMBED_MODEL,
            input=text
        )
        vec = response['data'][0]['embedding']

    arr = np.array(vec, dtype=np.float32)
    if arr.shape != (EMBED_DIM,):
        raise ValueError(f"Unexpected embedding shape {arr.shape}")
    return arr

def load_resources():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Missing FAISS index or metadata. Run embed_and_store.py first.")
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def search(query: str, k: int = 5) -> List[Tuple[int, float, Dict]]:
    index, metadata = load_resources()
    qvec = embed_query(query).reshape(1, -1)
    D, I = index.search(qvec, k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        meta = metadata.get(int(idx), {})
        results.append((int(idx), float(dist), meta))
    return results

if __name__ == "__main__":
    query = "What decisions were made in the August meetings?"
    hits = search(query, k=5)
    for i, (vid, dist, meta) in enumerate(hits, 1):
        print(f"{i}. ID={vid}  Distance={dist:.4f}  File={meta.get('filename')}  Chunk={meta.get('chunk_id')}")
        print(meta.get("text_preview", "")[:300], "\n---")
