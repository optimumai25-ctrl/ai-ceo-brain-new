import os
import io
import csv
import streamlit as st
import docx
import pandas as pd
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

# ─────────────────────────────────────────────────────────────
# ✅ Authentication from Streamlit secrets (service account)
# ─────────────────────────────────────────────────────────────
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
gdrive_secrets = st.secrets["gdrive"]
creds = service_account.Credentials.from_service_account_info(dict(gdrive_secrets), scopes=SCOPES)
service = build("drive", "v3", credentials=creds)

# ─────────────────────────────────────────────────────────────
# 🔧 Constants
# ─────────────────────────────────────────────────────────────
KB_FOLDER_NAME = "AI_CEO_KnowledgeBase"
REMINDERS_FOLDER_NAME = "AI_CEO_Reminders"  # NEW
OUTPUT_DIR = "parsed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOWTEXT_LOG = os.path.join(OUTPUT_DIR, "low_text_files.csv")

# ─────────────────────────────────────────────────────────────
# 🔍 Drive Helpers
# ─────────────────────────────────────────────────────────────
def get_folder_id_by_exact_name(folder_name):
    query = (
        f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' "
        f"and trashed = false"
    )
    results = service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name)',
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()
    folders = results.get('files', [])
    if not folders:
        raise Exception(f"Folder '{folder_name}' not found in Drive.")
    return folders[0]['id']

def list_folder_contents(parent_id):
    query = f"'{parent_id}' in parents and trashed = false"
    results = service.files().list(
        q=query,
        fields='files(id, name, mimeType)',
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()
    return results.get('files', [])

def download_file(file_id):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return fh

# ─────────────────────────────────────────────────────────────
# 📄 Extractors
# ─────────────────────────────────────────────────────────────
def extract_text_from_pdf(fh: io.BytesIO) -> str:
    # Prefer PyMuPDF
    data = fh.read()
    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for p in doc:
        pages.append(p.get_text("text") or "")
    text = "\n".join(pages)

    # Fallback to PyPDF2 if too little text
    if len(text.strip()) < 200:
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
    return text

def extract_text_from_docx(fh: io.BytesIO) -> str:
    doc = docx.Document(fh)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_excel(fh: io.BytesIO) -> str:
    df = pd.read_excel(fh)
    return df.to_string(index=False)

# ─────────────────────────────────────────────────────────────
# 🧾 Save parsed TXT with headers
# ─────────────────────────────────────────────────────────────
def write_parsed_output(folder_label: str, name: str, text: str):
    base_name = os.path.splitext(name)[0].replace(' ', '_')
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"[FOLDER]: {folder_label}\n[FILE]: {name}\n\n{text}")
    print(f"✅ Saved to {output_path}")

    # low-text heuristic
    if len(text.strip()) < 500:
        hdr_exists = os.path.exists(LOWTEXT_LOG)
        with open(LOWTEXT_LOG, "a", newline="", encoding="utf-8") as lf:
            w = csv.writer(lf)
            if not hdr_exists:
                w.writerow(["folder", "file", "chars"])
            w.writerow([folder_label, name, len(text.strip())])

# ─────────────────────────────────────────────────────────────
# 🧭 Process a single Drive file
# ─────────────────────────────────────────────────────────────
def process_and_save(file, folder_label):
    file_id = file['id']
    name = file['name']
    mime = file['mimeType']

    print(f"📄 Processing: {name}")
    try:
        if mime == 'application/pdf':
            fh = download_file(file_id)
            text = extract_text_from_pdf(fh)
        elif mime == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            fh = download_file(file_id)
            text = extract_text_from_docx(fh)
        elif mime == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            fh = download_file(file_id)
            text = extract_text_from_excel(fh)
        else:
            print(f"❌ Skipping unsupported file type: {name}")
            return
        write_parsed_output(folder_label, name, text)
    except Exception as e:
        print(f"❌ Error processing {name}: {e}")

# ─────────────────────────────────────────────────────────────
# ▶️ Main: scan KnowledgeBase subfolders + Reminders folder
# ─────────────────────────────────────────────────────────────
def parse_knowledgebase():
    """
    For AI_CEO_KnowledgeBase: treat each immediate subfolder as a label (e.g., Meetings/HR/Finance).
    Files at the root are ignored (to keep structure clean).
    """
    parent_id = get_folder_id_by_exact_name(KB_FOLDER_NAME)
    folders = list_folder_contents(parent_id)

    for folder in folders:
        if folder['mimeType'] != 'application/vnd.google-apps.folder':
            # Skip files placed at root of KnowledgeBase
            continue
        subfolder_label = folder['name']
        print(f"\n📁 Scanning KB subfolder: {subfolder_label}")
        subfolder_id = folder['id']
        files = list_folder_contents(subfolder_id)
        if not files:
            print("   (empty)")
            continue
        for file in files:
            process_and_save(file, subfolder_label)

def parse_reminders():
    """
    AI_CEO_Reminders: flat folder of small .txt/.docx/.pdf/.xlsx notes.
    We label all outputs with folder_label = 'Reminders'.
    """
    try:
        reminders_id = get_folder_id_by_exact_name(REMINDERS_FOLDER_NAME)
    except Exception:
        print(f"⚠️ Reminders folder '{REMINDERS_FOLDER_NAME}' not found; skipping.")
        return

    print(f"\n📁 Scanning Reminders folder: {REMINDERS_FOLDER_NAME}")
    files = list_folder_contents(reminders_id)
    if not files:
        print("   (empty)")
        return
    for file in files:
        process_and_save(file, "Reminders")

def main():
    print("🔎 Parsing Google Drive content into parsed_data/*.txt …")
    parse_knowledgebase()
    parse_reminders()
    print("✅ Parsing complete.")

if __name__ == '__main__':
    main()
