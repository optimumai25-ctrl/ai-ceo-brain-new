import os
import io
import streamlit as st
import docx
import pandas as pd
from PyPDF2 import PdfReader
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

# ─────────────────────────────────────────────────────────────
# ✅ Authentication from Streamlit secrets (gdrive2)
# ─────────────────────────────────────────────────────────────
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
gdrive_secrets = st.secrets["gdrive"]
creds = service_account.Credentials.from_service_account_info(dict(gdrive_secrets), scopes=SCOPES)
service = build("drive", "v3", credentials=creds)

# ─────────────────────────────────────────────────────────────
# 🔧 Constants
# ─────────────────────────────────────────────────────────────
FOLDER_NAME = 'AI_CEO_KnowledgeBase'
OUTPUT_DIR = 'parsed_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 🔍 Helpers
# ─────────────────────────────────────────────────────────────
def get_folder_id(folder_name):
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    folders = results.get('files', [])
    if not folders:
        raise Exception(f"Folder '{folder_name}' not found in Drive.")
    return folders[0]['id']

def list_folder_contents(parent_id):
    query = f"'{parent_id}' in parents"
    results = service.files().list(q=query, fields='files(id, name, mimeType)').execute()
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

def extract_text_from_pdf(fh):
    reader = PdfReader(fh)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def extract_text_from_docx(fh):
    doc = docx.Document(fh)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_excel(fh):
    df = pd.read_excel(fh)
    return df.to_string(index=False)

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

        base_name = os.path.splitext(name)[0].replace(' ', '_')
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"[FOLDER]: {folder_label}\n[FILE]: {name}\n\n{text}")
        print(f"✅ Saved to {output_path}")

    except Exception as e:
        print(f"❌ Error processing {name}: {e}")

# ─────────────────────────────────────────────────────────────
# ▶️ Main
# ─────────────────────────────────────────────────────────────
def main():
    parent_id = get_folder_id(FOLDER_NAME)
    folders = list_folder_contents(parent_id)

    for folder in folders:
        if folder['mimeType'] != 'application/vnd.google-apps.folder':
            continue  # Skip files at root
        print(f"\n📁 Scanning folder: {folder['name']}")
        subfolder_id = folder['id']
        files = list_folder_contents(subfolder_id)
        if not files:
            print("   (empty)")
        for file in files:
            process_and_save(file, folder['name'])

if __name__ == '__main__':
    main()
