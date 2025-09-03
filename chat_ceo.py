import json
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd

# Local modules
import file_parser
import embed_and_store
from answer_with_rag import answer

# Drive uploader utilities (already configured to use st.secrets)
from gdrive_uploader import find_or_create_folder, upload_or_update_file, service

# ──────────────────────────────────
# App Config
# ──────────────────────────────────
st.set_page_config(page_title="AI CEO Assistant", page_icon="🧠", layout="wide")

# Hardcoded demo login (replace with OAuth if needed)
USERNAME = "admin123"
PASSWORD = "BestOrg123@#"

# Paths
HIST_PATH = Path("chat_history.json")
REFRESH_PATH = Path("last_refresh.txt")

# Drive folders
REMINDERS_FOLDER = "AI_CEO_Reminders"        # new folder that stores reminders/notes
KB_FOLDER = "AI_CEO_KnowledgeBase"           # existing knowledge base (unchanged)

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

def save_reminder_to_drive(content: str, title_hint: str = "") -> str:
    """
    Save a reminder/note as a small .txt in AI_CEO_Reminders.
    Returns the file name used for upload.
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    base_name = f"{ts}_Reminder"
    if title_hint:
        # sanitize to keep it Drive-friendly
        safe = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in title_hint.strip())
        safe = "_".join(safe.split())[:60]
        if safe:
            base_name += f"_{safe}"
    fname = f"{base_name}.txt"

    folder_id = find_or_create_folder(service, REMINDERS_FOLDER)
    with open(fname, "w", encoding="utf-8") as f:
        f.write(content)
    try:
        upload_or_update_file(service, fname, folder_id)
    finally:
        # local temp file cleanup
        try:
            Path(fname).unlink(missing_ok=True)
        except Exception:
            pass
    return fname

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
    ["💬 New Chat", "📝 New Reminder", "📜 View History", "🔁 Refresh Data"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Reminders are saved to Google Drive → AI_CEO_Reminders.")

# ──────────────────────────────────
# Modes
# ──────────────────────────────────

# 1) Refresh (Parse + Embed) both KnowledgeBase and Reminders
if mode == "🔁 Refresh Data":
    st.title("🔁 Refresh AI Knowledge Base")
    st.caption("Re-parse Google Drive documents (KnowledgeBase + Reminders) and re-embed vectors.")
    st.markdown(f"🧓 **Last Refreshed:** {load_refresh_time()}")

    if st.button("🚀 Run File Parser + Embedder"):
        with st.spinner("Refreshing knowledge base..."):
            try:
                # Parse: this will scan AI_CEO_KnowledgeBase and AI_CEO_Reminders
                file_parser.main()
                # Embed
                embed_and_store.main()
                save_refresh_time()
                st.success("✅ Data refreshed and embedded successfully.")
                st.markdown(f"🧓 **Last Refreshed:** {load_refresh_time()}")
            except Exception as e:
                st.error(f"❌ Failed: {e}")

# 2) Create a reminder manually (without chat)
elif mode == "📝 New Reminder":
    st.title("📝 Save a Reminder / Note")
    st.caption("This will be stored in Google Drive → AI_CEO_Reminders and included after the next Refresh.")
    with st.form("new_reminder_form"):
        title = st.text_input("Short title (optional)")
        body = st.text_area("Reminder content", height=200, placeholder="Write a fact, decision, SOP step, or any short note...")
        submitted = st.form_submit_button("💾 Save to Drive")
        if submitted:
            if body.strip():
                fname = save_reminder_to_drive(body.strip(), title_hint=title)
                st.success(f"✅ Saved reminder as **{fname}** to Drive/AI_CEO_Reminders.")
                st.info("Run **Refresh Data** to parse and embed the new reminder.")
            else:
                st.warning("Please write some content before saving.")

# 3) History view
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

# 4) Chat interface (with “Save as Reminder” after replies)
elif mode == "💬 New Chat":
    st.title("🧠 AI CEO Assistant")
    st.caption("Ask about meetings, projects, hiring, finances. If the answer lacks context, click 'Save as Reminder' to teach the system.")
    st.markdown(f"🧓 **Last Refreshed:** {load_refresh_time()}")

    # Optional retrieval scope toggle
    limit_meetings = st.checkbox("Limit retrieval to Meetings", value=True)

    history = load_history()
    for turn in history:
        with st.chat_message(turn.get("role", "assistant")):
            st.markdown(f"**[{turn.get('timestamp', 'N/A')}]**  \n{turn.get('content', '')}")

    user_msg = st.chat_input("Type your question…")
    if user_msg:
        now = datetime.now().strftime('%b-%d-%Y %I:%M%p')
        history.append({
            "role": "user",
            "content": user_msg,
            "timestamp": now
        })

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    reply = answer(user_msg, k=7, chat_history=history, restrict_to_meetings=limit_meetings)
                except Exception as e:
                    reply = f"Error: {e}"
            ts = datetime.now().strftime('%b-%d-%Y %I:%M%p')
            st.markdown(f"**[{ts}]**  \n{reply}")

            # ——— Save-as-Reminder UI (always available) ———
            with st.expander("💾 Save this as a Reminder"):
                default_note = f"Q: {user_msg}\n\nA: {reply}\n\n(Note created on {ts})"
                note = st.text_area("Reminder content to save in Drive:", value=default_note, height=180)
                col1, col2 = st.columns([1,1])
                with col1:
                    title_hint = st.text_input("Optional short title", value="")
                with col2:
                    if st.button("Save to Drive → AI_CEO_Reminders"):
                        if note.strip():
                            fname = save_reminder_to_drive(note.strip(), title_hint=title_hint)
                            st.success(f"✅ Saved reminder as **{fname}**. It will be indexed after the next Refresh.")
                        else:
                            st.warning("Please provide some content to save.")

        history.append({
            "role": "assistant",
            "content": reply,
            "timestamp": ts
        })
        save_history(history)
