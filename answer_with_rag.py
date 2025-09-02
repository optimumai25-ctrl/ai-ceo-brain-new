from typing import List, Dict, Optional
from semantic_search import search

# OpenAI client setup
try:
    from openai import OpenAI
    client = OpenAI()
    use_client = True
except Exception:
    import openai
    import os
    openai.api_key = os.getenv("OPENAI_API_KEY")
    use_client = False

COMPLETIONS_MODEL = "gpt-4o"
MAX_CONTEXT_CHARS = 8000

# --- NEW: meeting index support ---
from meeting_indexer import find_tasks_by_person, list_attendees, load_index
import re
from datetime import datetime

def _find_date_in_query(q: str) -> Optional[str]:
    # Accepts 'on Sep 02', 'on September 2, 2025', 'on 2025-09-02'
    q = q.strip()
    m = re.search(r"(\d{4}-\d{2}-\d{2})", q)
    if m:
        return m.group(1)
    m = re.search(r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})(?:,\s*(\d{4}))?", q, flags=re.I)
    if m:
        mon, day, year = m.group(1), int(m.group(2)), m.group(3)
        mon = mon[:3].title()
        if year:
            y = int(year)
        else:
            idx = load_index()
            years = sorted({(d or "0000")[:4] for d in [mt.get("date") for mt in idx.get("meetings", [])] if d})
            y = int(years[-1]) if years else datetime.now().year
        try:
            dt = datetime.strptime(f"{mon} {day} {y}", "%b %d %Y")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None
    return None

def build_context(hits):
    ctx, total = [], 0
    for _, _, meta in hits:
        t = meta.get("text_preview", "")
        if not t:
            continue
        if total + len(t) > MAX_CONTEXT_CHARS:
            break
        ctx.append(t); total += len(t)
    return "\n\n---\n\n".join(ctx)

def ask_gpt(query, context, chat_history: List[Dict]):
    messages = [{"role": "system", "content": "You are a precise executive assistant. Use the provided context verbatim for facts and cite file names when possible."}]
    for m in chat_history[-6:]:
        messages.append({"role": m["role"], "content": m["content"]})
    if context:
        messages.append({"role": "system", "content": f"Context:\n{context}"})
    messages.append({"role": "user", "content": query})

    if use_client:
        resp = client.chat.completions.create(model=COMPLETIONS_MODEL, messages=messages, temperature=0.2)
        return resp.choices[0].message.content
    else:
        resp = openai.ChatCompletion.create(model=COMPLETIONS_MODEL, messages=messages, temperature=0.2)
        return resp.choices[0].message["content"]

def _try_people_router(query: str) -> Optional[str]:
    q = query.strip()

    # 1) "Who attended ... [on <date>]"
    if re.search(r"\bwho\b.*\b(attended|were (?:there|present)|participants?)\b", q, flags=re.I):
        d = _find_date_in_query(q)
        attendees = list_attendees(date=d)
        if not attendees:
            return "No attendees found in the indexed meetings for that date." if d else "No attendees found in the indexed meetings."
        return (f"Attendees on {d}: " if d else "People seen across meetings: ") + ", ".join(attendees)

    # 2) "Tasks/Action Items/Next Steps for <Name> [on <date>]"
    m = re.search(r"\b(tasks?|action items?|next steps?)\s+(?:for|assigned to)\s+([A-Z][\w.'-]+)\b", q, flags=re.I)
    if not m:
        m = re.search(r"\bwhat\s+did\s+([A-Z][\w.'-]+)\b.*\b(do|commit|own|deliver)\b", q, flags=re.I)
    if m:
        name = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1)
        d = _find_date_in_query(q)
        rows = find_tasks_by_person(name, date=d)
        if not rows:
            return f"No tasks found for {name}" + (f" on {d}." if d else ".")
        lines = [f"• {r['task']}  [{r.get('date') or 'n/a'} {r.get('time') or ''} · {r.get('file') or ''}]".strip() for r in rows]
        return f"Tasks for {name}" + (f" on {d}" if d else " (all meetings)") + ":\n" + "\n".join(lines)

    # 3) A single capitalized token → treat as name lookup if that person exists
    idx = load_index()
    known = set(idx.get("people_index", {}).keys())
    tok = re.match(r"^\s*([A-Z][\w.'-]+)\s*[:,-]?\s*$", q)
    if tok:
        nm = tok.group(1)
        if nm in known:
            rows = find_tasks_by_person(nm, date=None)
            if not rows:
                return f"No tasks found for {nm}."
            lines = [f"• {r['task']}  [{r.get('date') or 'n/a'} {r.get('time') or ''} · {r.get('file') or ''}]".strip() for r in rows]
            return f"Tasks for {nm} (all meetings):\n" + "\n".join(lines)

    return None

def answer(query: str, k: int = 5, chat_history: List[Dict] = []) -> str:
    routed = _try_people_router(query)
    if routed:
        return routed
    hits = search(query, k=k)
    if not hits:
        return ask_gpt(query, context="", chat_history=chat_history)
    context = build_context(hits)
    return ask_gpt(query, context=context, chat_history=chat_history)

if __name__ == "__main__":
    from chat_ceo import load_history
    print(answer("Who attended the meeting on September 02, 2025?", chat_history=load_history()))
    print(answer("Tasks for Sai on 2025-08-26", chat_history=load_history()))

