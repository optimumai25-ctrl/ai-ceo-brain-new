import os
import io
from pathlib import Path
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
import docx  # python-docx
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

# ─────────────────────────────────────────────────────────────
# Auth (from Streamlit secrets)
# ─────────────────────────────────────────────────────────────
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
gdrive_secrets = st.secrets["gdrive"]
creds = service_account.Credentials.from_service_account_info(
    dict(gdrive_secrets), scopes=SCOPES
)
service = build("drive", "v3", credentials=creds)

SHARED_DRIVE_ID = gdrive_secrets.get("shared_drive_id")  # optional

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
FOLDER_NAME = "AI_CEO_KnowledgeBase"
OUTPUT_DIR = Path("parsed_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Helpers: Drive list / export / download
# ─────────────────────────────────────────────────────────────
def _drive_list(**kwargs):
    # Adds All-Drives support when shared drive id is provided
    base = dict(
        fields="nextPageToken, files(id, name, mimeType)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
        pageSize=1000,
    )
    if SHARED_DRIVE_ID:
        base.update(corpora="drive", driveId=SHARED_DRIVE_ID)
    base.update(kwargs)
    return service.files().list(**base)

def get_folder_id_by_name(folder_name: str) -> str:
    q = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    resp = _drive_list(q=q).execute()
    folders = resp.get("files", [])
    if not folders:
        raise RuntimeError(f"Folder '{folder_name}' not found.")
    return folders[0]["id"]

def iter_children(parent_id: str):
    q = f"'{parent_id}' in parents and trashed=false"
    page_token = None
    while True:
        resp = _drive_list(q=q, pageToken=page_token).execute()
        for f in resp.get("files", []):
            yield f
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

def download_bytes(file_id: str) -> io.BytesIO:
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return fh

def export_google_doc_text(file_id: str) -> str:
    data = service.files().export(
        fileId=file_id, mimeType="text/plain"
    ).execute()
    return data.decode("utf-8", errors="ignore")

def export_google_sheet_csv_bytes(file_id: str) -> bytes:
    return service.files().export(
        fileId=file_id, mimeType="text/csv"
    ).execute()

# ─────────────────────────────────────────────────────────────
# Helpers: Extraction by type
# ─────────────────────────────────────────────────────────────
def extract_text_from_pdf_bytes(fh: io.BytesIO) -> str:
    reader = PdfReader(fh)
    return "\n".join((page.extract_text() or "") for page in reader.pages)

def extract_text_from_docx_bytes(fh: io.BytesIO) -> str:
    doc = docx.Document(fh)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text_from_excel_bytes(fh: io.BytesIO) -> str:
    df = pd.read_excel(fh)  # xlsx (openpyxl). For .xls add xlrd in requirements.
    return df.to_string(index=False)

def extract_text_from_csv_bytes(b: bytes) -> str:
    # Try UTF-8, fallback to latin-1
    try:
        df = pd.read_csv(io.BytesIO(b))
    except Exception:
        df = pd.read_csv(io.BytesIO(b), encoding="latin1")
    return df.to_string(index=False)

# ─────────────────────────────────────────────────────────────
# Core processing
# ─────────────────────────────────────────────────────────────
def process_file(file_obj: dict, path_label: str):
    fid = file_obj["id"]
    name = file_obj["name"]
    mime = file_obj["mimeType"]

    text = None

    # Google-native types (export)
    if mime == "application/vnd.google-apps.document":
        text = export_google_doc_text(fid)
    elif mime == "application/vnd.google-apps.spreadsheet":
        csv_bytes = export_google_sheet_csv_bytes(fid)
        text = extract_text_from_csv_bytes(csv_bytes)

    # Regular uploaded files (download and parse)
    elif mime == "application/pdf":
        text = extract_text_from_pdf_bytes(download_bytes(fid))
    elif mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_docx_bytes(download_bytes(fid))
    elif mime == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        text = extract_text_from_excel_bytes(download_bytes(fid))
    elif mime == "text/csv":
        text = extract_text_from_csv_bytes(download_bytes(fid).read())

    # Optional: legacy Excel (.xls). Requires xlrd.
    elif mime == "application/vnd.ms-excel":
        try:
            text = pd.read_excel(download_bytes(fid), engine="xlrd").to_string(index=False)
        except Exception as e:
            print(f"Skipping legacy Excel (.xls) without xlrd: {name} ({e})")
            return

    else:
        print(f"Skipping unsupported type: {name} [{mime}]")
        return

    # Persist normalized .txt for embedding
    stem = os.path.splitext(name)[0].replace(" ", "_")
    out = OUTPUT_DIR / f"{stem}.txt"
    header = f"[PATH]: {path_label}\n[FILE]: {name}\n\n"
    out.write_text(header + (text or ""), encoding="utf-8")
    print(f"Saved: {out}")

def walk_folder(folder_id: str, path_label: str):
    for item in iter_children(folder_id):
        if item["mimeType"] == "application/vnd.google-apps.folder":
            child_path = f"{path_label}/{item['name']}" if path_label else item["name"]
            walk_folder(item["id"], child_path)
        else:
            process_file(item, path_label or "/")

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    root_id = get_folder_id_by_name(FOLDER_NAME)
    # Process files at the root folder itself
    for item in iter_children(root_id):
        if item["mimeType"] == "application/vnd.google-apps.folder":
            walk_folder(item["id"], item["name"])
        else:
            process_file(item, "/")

if __name__ == "__main__":
    main()

