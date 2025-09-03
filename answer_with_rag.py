from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import re

from semantic_search import (
    search,
    search_meetings,
    search_in_date_window,
)

# OpenAI client setup (new SDK preferred, fallback legacy)
try:
    from openai import OpenAI
    _client = OpenAI()
    _use_client = True
except Exception:
    _client = None
    _use_client = False
    import openai  # type: ignore
    import os
    openai.api_key = os.getenv("OPENAI_API_KEY")

COMPLETIONS_MODEL = "gpt-4o"
MAX_CONTEXT_CHARS = 8000

def build_context(topk: List[Tuple[int, float, Dict]]) -> str:
    parts, total = [], 0
    for _, _, meta in topk:
        fname = meta.get("filename", "unknown.txt")
        cid = meta.get("chunk_id", 0)
        text = meta.get("text_preview", "")
        snippet = f"[SOURCE: {fname} | CHUNK: {cid}]\n{text}\n"
        if total + len(snippet) > MAX_CONTEXT_CHARS:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n".join(parts)

_MONTHS = "(january|february|march|april|may|june|july|august|september|october|november|december)"

def resolve_date_window_from_query(q: str):
    s = q.lower()
    today = datetime.now()

    if "last week" in s:
        weekday = today.weekday()
        end = today - timedelta(days=weekday + 1)   # last Sunday
        start = end - timedelta(days=6)             # last Monday
        return (start.replace(hour=0, minute=0, second=0, microsecond=0),
                end.replace(hour=23, minute=59, second=59, microsecond=0))

    if "last month" in s:
        first_this = today.replace(day=1)
        last_prev = first_this - timedelta(days=1)
        first_prev = last_prev.replace(day=1)
        return (first_prev.replace(hour=0, minute=0, second=0, microsecond=0),
                last_prev.replace(hour=23, minute=59, second=59, microsecond=0))

    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', s)
    if m:
        y, mo, d = map(int, m.groups())
        start = datetime(y, mo, d, 0, 0, 0)
        end   = datetime(y, mo, d, 23, 59, 59)
        return (start, end)

    m2 = re.search(rf'{_MONTHS}\s+(\d{{1,2}}),\s*(\d{{4}})', s, re.I)
    if m2:
        month_name, dd, yy = m2.groups()
        dt = datetime.strptime(f"{month_name} {dd} {yy}", "%B %d %Y")
        start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end   = dt.replace(hour=23, minute=59, second=59, microsecond=0)
        return (start, end)

    return None

def ask_gpt(query: str, context: str = "", chat_history: List[Dict] = []) -> str:
    system = (
        "You are a precise Virtual CEO assistant. "
        "Use provided sources and cite [filename#chunk] like [2025-09-02_Meeting-Summary.txt#2]. "
        "If no sources are provided, answer briefly with general knowledge."
    )
    messages: List[Dict] = [{"role": "system", "content": system}]
    for msg in chat_history[-4:]:
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")
        role = msg.get("role", "user")
        formatted = f"[{timestamp}] {content}" if timestamp else content
        messages.append({"role": role, "content": formatted})

    if context:
        messages.append({"role": "user", "content": f"Query:\n{query}\n\nSources:\n{context}"})
    else:
        messages.append({"role": "user", "content": query})

    if _use_client:
        resp = _client.chat.completions.create(  # type: ignore
            model=COMPLETIONS_MODEL,
            messages=messages[-6:],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    else:
        resp = openai.ChatCompletion.create(  # type: ignore
            model=COMPLETIONS_MODEL,
            messages=messages[-6:],
            temperature=0.2,
        )
        return resp.choices[0].message["content"]

def answer(query: str, k: int = 5, chat_history: List[Dict] = [], restrict_to_meetings: bool = False) -> str:
    win = resolve_date_window_from_query(query)
    if win:
        start, end = win
        hits = search_in_date_window(query, start, end, k=k)
    else:
        hits = search_meetings(query, k=k) if restrict_to_meetings else search(query, k=k)

    if not hits:
        return ask_gpt(query, context="", chat_history=chat_history)

    ctx = build_context(hits)
    return ask_gpt(query, context=ctx, chat_history=chat_history)

if __name__ == "__main__":
    print(answer("Summarize decisions from last week.", k=7))

