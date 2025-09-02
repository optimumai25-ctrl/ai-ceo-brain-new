import os
import time
import pickle
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import faiss
from dotenv import load_dotenv
from tqdm import tqdm

from chunk_utils import simple_chunks

# -------- Load OpenAI API Key from .env --------
import openai
load_dotenv()  # ✅ Load variables from .env
import streamlit as st
import openai

openai.api_key = st.secrets["OPENAI_API_KEY"]

# -------- Paths & Config --------
PARSED_DIR = Path("parsed_data")
EMBED_DIR = Path("embeddings")
EMBED_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
INDEX_PATH = EMBED_DIR / "faiss.index"
META_PATH = EMBED_DIR / "metadata.pkl"

# FAISS index
base_index = faiss.IndexFlatL2(EMBED_DIM)
index = faiss.IndexIDMap2(base_index)

metadata: Dict[int, Dict] = {}
next_id = 0

# -------- Embedding --------
def get_embedding(text: str) -> Optional[np.ndarray]:
    for attempt in range(4):
        try:
            response = openai.Embedding.create(
                model=EMBED_MODEL,
                input=text
            )
            vec = response["data"][0]["embedding"]
            arr = np.array(vec, dtype=np.float32)
            if arr.shape != (EMBED_DIM,):
                raise ValueError(f"Unexpected embedding shape {arr.shape}")
            return arr
        except Exception as e:
            wait = 1.5 ** attempt
            print(f"Embedding error (attempt {attempt + 1}): {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    print("Failed to embed after retries.")
    return None

def add_to_index(vec: np.ndarray, vid: int):
    index.add_with_ids(vec.reshape(1, -1), np.array([vid], dtype=np.int64))

# -------- Main --------
def main():
    global next_id
    if not PARSED_DIR.exists():
        print(f"Missing folder: {PARSED_DIR.resolve()}")
        return

    files = sorted([p for p in PARSED_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])
    if not files:
        print("No .txt files found in parsed_data.")
        return

    print(f"Found {len(files)} files to embed (chunking enabled).")
    for fp in tqdm(files, desc="Embedding"):
        text = fp.read_text(encoding="utf-8").strip()
        if not text:
            print(f"Skipping empty: {fp.name}")
            continue

        chunks = simple_chunks(text, max_chars=3500, overlap=300)
        if not chunks:
            chunks = [{"chunk_id": 0, "text": text[:3500]}]

        for ch in chunks:
            vec = get_embedding(ch["text"])
            if vec is None:
                print(f"Skipping chunk {ch['chunk_id']} of {fp.name} due to embedding failure.")
                continue
            add_to_index(vec, next_id)
            metadata[next_id] = {
                "filename": fp.name,
                "path": str(fp),
                "chunk_id": ch["chunk_id"],
                "text_preview": ch["text"][:1000]
            }
            next_id += 1

    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Saved FAISS index to {INDEX_PATH}")
    print(f"✅ Saved metadata for {len(metadata)} vectors to {META_PATH}")

if __name__ == "__main__":
    main()
