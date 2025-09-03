# chat_ceo.py
import json
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd

import file_parser
import embed_and_store
from answer_with_rag import answer

from reminders_extractor import extract_from_csv

st.set_page_config(page_title="AI CEO Assistant", page_icon="🧠", layout="wide")

USERNAME = "admin123"
PASSWORD = "BestOrg123@#"

HIST_PATH = Path("chat_history.json")
REFRESH_PATH = Path("last_refresh.txt")

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

st.sidebar.title("🧠 AI CEO Panel")
st.sidebar.markdown(f"👤 Logged in as: `{USERNAME}`")

if st.sidebar.button("🔓 Logout"):
    st.session_state["authenticated"] = False
    st.rerun()

mode = st.sidebar.radio(
    "Navigation",
    ["💬 New Chat", "📤 Upload Chat CSV (extract REMINDERs)", "📜 View History", "🔁 Refresh Data"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Use 'REMINDER:' at the start of a message to teach the assistant.")

if mode == "🔁 Refresh Data":
    st.title("🔁 Refresh AI Knowledge Base")
    st.caption("Re-parse local reminders + (optional) Google Drive docs, then re-embed.")
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

elif mode == "📤 Upload Chat CSV (extract REMINDERs)":
    st.title("📤 Upload Chat History CSV")
    st.caption("This will extract rows starting with 'REMINDER:' into reminders/*.txt")
    up = st.file_uploader("Choose chat_history CSV", type=["csv"])
    if up:
        tmp_path = Path("uploaded_chat_history.csv")
        tmp_path.write_bytes(up.read())
        try:
            files = extract_from_csv(str(tmp_path))
            st.success(f"Extracted {len(files)} reminders into ./reminders")
            st.info("Run **Refresh Data** to index them.")
        except Exception as e:
            st.error(f"Extraction failed: {e}")
        finally:
            try: tmp_path.unlink(missing_ok=True)
            except Exception: pass

elif mode == "📜 View History":
    st.title("📜 Chat History")
    history = load_history()
    if not history:
        st.info("No chat history found.")
    else:
        for turn in history:
            role = "👤 You" if turn.get("role") == "user" else "🧠 Assistant"
            timestamp = turn.get("timestamp", "N/A")
            st.markdown(f"**{role} | [{timestamp}]**  \n{turn.get('content', '')}")

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
    st.caption("Ask about meetings, projects, policies. Add REMINDERs via CSV to teach the assistant.")
    st.markdown(f"🧓 **Last Refreshed:** {load_refresh_time()}")
    limit_meetings = st.checkbox("Limit retrieval to Meetings", value=True)

    history = load_history()
    for turn in history:
        with st.chat_message(turn.get("role", "assistant")):
            st.markdown(f"**[{turn.get('timestamp', 'N/A')}]**  \n{turn.get('content', '')}")

    user_msg = st.chat_input("Type your question…")
    if user_msg:
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

