import json
import re
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

import file_parser
import embed_and_store
from answer_with_rag import answer

# ──────────────────────────────────
# App Config
# ──────────────────────────────────
st.set_page_config(page_title="AI CEO Assistant", page_icon="🧠", layout="wide")

# Demo login (replace with OAuth if needed)
USERNAME = "admin123"
PASSWORD = "BestOrg123@#"

# Paths
HIST_PATH = Path("chat_history.json")
REFRESH_PATH = Path("last_refresh.txt")

# ──────────────────────────────────
# Auth
# ──────────────────────────────────
def login():
    st.title("🔐 Login to AI CEO Assistant")
    with st.form("login_form"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if u == USERNAME and p == PASSWORD:
                st.session_state["authenticated"] = True
                st.success("Login successful.")
                st.rerun()
            else:
                st.error("Invalid username or password.")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if not st.session_state["authenticated"]:
    login()
    st.stop()

# ──────────────────────────────────
# Helpers
# ──────────────────────────────────
def load_history():
    if HIST_PATH.exists():
        return json.loads(HIST_PATH.read_text(encoding="utf-8"))
    return []

def save_history(history):
    HIST_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

def reset_chat():
    if HIST_PATH.exists():
        HIST_PATH.unlink()

def save_refresh_time():
    REFRESH_PATH.write_text(datetime.now().strftime('%b-%d-%Y %I:%M %p'))

def load_refresh_time():
    if REFRESH_PATH.exists():
        return REFRESH_PATH.read_text()
    return "Never"

def export_history_to_csv(history: list) -> bytes:
    df = pd.DataFrame(history)
    return df.to_csv(index=False).encode('utf-8')

def save_reminder_local(content: str, title_hint: str = "") -> str:
    """
    Save a REMINDER as a structured .txt in ./reminders and return the file path.
    Accepts either a plain sentence or a structured block with Title/Tags/ValidFrom/Body.
    """
    Path("reminders").mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    title = title_hint.strip() or (content.strip().split("\n", 1)[0][:60] or "Untitled")
    safe_title = re.sub(r"[^A-Za-z0-9_\-]+", "_", title)
    fp = Path("reminders") / f"{ts}_{safe_title}.txt"

    is_structured = bool(re.search(r'(?mi)^\s*Title:|^\s*Tags:|^\s*ValidFrom:|^\s*Body:', content))
    if is_structured:
        payload = content.strip() + "\n"
    else:
        payload = (
            f"Title: {title}\n"
            f"Tags: reminder\n"
            f"ValidFrom: {datetime.now():%Y-%m-%d}\n"
            f"Body: {content.strip()}\n"
        )
    fp.write_text(payload, encoding="utf-8")
    return str(fp)

# ──────────────────────────────────
# Sidebar
# ──────────────────────────────────
st.sidebar.title("🧠 AI CEO Panel")
st.sidebar.markdown(f"👤 Logged in as: `{USERNAME}`")
if st.sidebar.button("🔓 Logout"):
    st.session_state["authenticated"] = False
    st.rerun()

mode = st.sidebar.radio(
    "Navigation",
    ["💬 New Chat", "📜 View History", "🔁 Refresh Data"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Tip: Start a message with **REMINDER:** to teach the assistant instantly.")

# ──────────────────────────────────
# Modes
# ──────────────────────────────────

if mode == "🔁 Refresh Data":
    st.title("🔁 Refresh AI Knowledge Base")
    st.caption("Parses local reminders + (optional) Google Drive docs, then re-embeds.")
    st.markdown(f"🧓 **Last Refreshed:** {load_refresh_time()}")

    if st.button("🚀 Run File Parser + Embedder"):
        with st.spinner("Refreshing knowledge base..."):
            try:
                file_parser.main()
                embed_and_store.main()
                save_refresh_time()
                st.success("✅ Data refreshed and embedded successfully.")
                st.markdown(f"🧓 **Last Refreshed:** {load_refresh_time()}")
            except Exception as e:
                st.error(f"❌ Failed: {e}")

elif mode == "📜 View History":
    st.title("📜 Chat History")
    history = load_history()
    if not history:
        st.info("No chat history found.")
    else:
        for turn in history:
            role = "👤 You" if turn.get("role") == "user" else "🧠 Assistant"
            ts = turn.get("timestamp", "N/A")
            st.markdown(f"**{role} | [{ts}]**  \n{turn.get('content', '')}")

        st.markdown("---")
        st.download_button(
            label="⬇️ Download Chat History as CSV",
            data=export_history_to_csv(history),
            file_name="chat_history.csv",
            mime="text/csv"
        )
        if st.button("🗑️ Clear Chat History"):
            reset_chat()
            st.success("History cleared.")

elif mode == "💬 New Chat":
    st.title("🧠 AI CEO Assistant")
    st.caption("Ask about meetings, projects, policies. Start a message with REMINDER: to teach facts.")
    st.markdown(f"🧓 **Last Refreshed:** {load_refresh_time()}")
    limit_meetings = st.checkbox("Limit retrieval to Meetings", value=True)

    history = load_history()
    for turn in history:
        with st.chat_message(turn.get("role", "assistant")):
            st.markdown(f"**[{turn.get('timestamp', 'N/A')}]**  \n{turn.get('content', '')}")

    user_msg = st.chat_input("Type your question or add a REMINDER…")
    if user_msg:
        # 1) If this is a REMINDER, save it immediately to ./reminders
        if user_msg.strip().lower().startswith("reminder:"):
            body = re.sub(r"^reminder:\s*", "", user_msg.strip(), flags=re.I)
            title_hint = body.split("\n", 1)[0][:60]
            saved_path = save_reminder_local(body, title_hint=title_hint)
            st.success(f"💾 Reminder saved: {saved_path}. Run '🔁 Refresh Data' to index it.")

        # 2) Normal chat flow
        now = datetime.now().strftime('%b-%d-%Y %I:%M%p')
        history.append({"role": "user", "content": user_msg, "timestamp": now})

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    reply = answer(user_msg, k=7, chat_history=history, restrict_to_meetings=limit_meetings)
                except Exception as e:
                    reply = f"Error: {e}"
            ts = datetime.now().strftime('%b-%d-%Y %I:%M%p')
            st.markdown(f"**[{ts}]**  \n{reply}")

        history.append({"role": "assistant", "content": reply, "timestamp": ts})
        save_history(history)
