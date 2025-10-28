from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import base64
import json
from urllib.parse import quote
from typing import Optional

# -------------------------------
# Config
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_files", "files")
os.makedirs(UPLOAD_DIR, exist_ok=True)

HTTP_PORT = int(os.getenv("FILE_DOWNLOAD_PORT", 8003))
HOST = os.getenv("HOST", "localhost")
BASE_DOWNLOAD_URL = f"http://{HOST}:{HTTP_PORT}/files"

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="FileServer API", version="1.0.0")

# Allow cross-origin (optional, useful if multiple agents on different hosts)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Endpoints
# -------------------------------

@app.post("/upload")
async def upload_file(filename: str = Form(...), content_base64: str = Form(...)):
    """
    Accept base64 encoded file and save to disk.
    """
    try:
        file_token = str(uuid.uuid4())
        file_ext = os.path.splitext(filename)[1]
        storage_name = f"{file_token}{file_ext}"
        filepath = os.path.join(UPLOAD_DIR, storage_name)

        data = base64.b64decode(content_base64)
        with open(filepath, "wb") as f:
            f.write(data)

        download_url = f"{BASE_DOWNLOAD_URL}/{quote(storage_name)}"

        return {
            "token": file_token,
            "download_url": download_url,
            "stored_name": storage_name,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/download/{token}")
async def download_file(token: str):
    """
    Download a file by token.
    Returns the file as base64-encoded string.
    """
    try:
        for fname in os.listdir(UPLOAD_DIR):
            if fname.startswith(token):
                filepath = os.path.join(UPLOAD_DIR, fname)
                with open(filepath, "rb") as f:
                    return {"filename": fname, "content_base64": base64.b64encode(f.read()).decode("utf-8")}
        raise HTTPException(status_code=404, detail=f"File with token {token} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/url/{token}")
async def get_download_url(token: str):
    """
    Get a public URL for a file by token.
    """
    try:
        for fname in os.listdir(UPLOAD_DIR):
            if fname.startswith(token):
                return {"download_url": f"{BASE_DOWNLOAD_URL}/{quote(fname)}"}
        raise HTTPException(status_code=404, detail=f"File with token {token} not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files/{filename}")
async def serve_file(filename: str):
    """
    Serve the actual file over HTTP for download.
    """
    filepath = os.path.join(UPLOAD_DIR, filename)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(filepath, filename=filename)


# -------------------------------
# Run with: uvicorn file_server:app --host 0.0.0.0 --port 8000
# -------------------------------
