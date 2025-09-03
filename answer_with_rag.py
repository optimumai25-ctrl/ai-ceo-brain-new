from typing import List, Dict
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

def build_context(topk: List[Dict]) -> str:
    parts, total = [], 0
    for r in topk:
        fname = r[2].get("filename", "unknown.txt")
        cid = r[2].get("chunk_id", 0)
        text = r[2].get("text_preview", "")
        snippet = f"[SOURCE: {fname} | CHUNK: {cid}]\n{text}\n"
        if total + len(snippet) > MAX_CONTEXT_CHARS:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n".join(parts)

def ask_gpt(query: str, context: str = "", chat_history: List[Dict] = []) -> str:
    system = (
        "You are a smart Virtual CEO assistant. If sources are provided, answer using them and cite by filename and chunk like [CEO_Notes.txt#2]. "
        "If no sources are provided, use your general knowledge."
    )

    messages = [{"role": "system", "content": system}]

    # Include up to last 4 chat history turns
    for msg in chat_history[-4:]:
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")
        role = msg.get("role", "user")
        formatted = f"[{timestamp}] {content}" if timestamp else content
        messages.append({"role": role, "content": formatted})

    if context:
        messages.append({
            "role": "user",
            "content": f"Query:\n{query}\n\nSources:\n{context}"
        })
    else:
        messages.append({"role": "user", "content": query})

    # Call OpenAI ChatCompletion
    if use_client:
        resp = client.chat.completions.create(
            model=COMPLETIONS_MODEL,
            messages=messages,
            temperature=0.2,
        )
        return resp.choices[0].message.content
    else:
        resp = openai.ChatCompletion.create(
            model=COMPLETIONS_MODEL,
            messages=messages,
            temperature=0.2,
        )
        return resp.choices[0].message["content"]

def answer(query: str, k: int = 5, chat_history: List[Dict] = []) -> str:
    hits = search(query, k=k)
    if not hits:
        return ask_gpt(query, context="", chat_history=chat_history)
    context = build_context(hits)
    return ask_gpt(query, context=context, chat_history=chat_history)

# Optional CLI test
if __name__ == "__main__":
    from chat_ceo import load_history
    print(answer("What are the goals for Q3 based on CEO notes?", chat_history=load_history()))
