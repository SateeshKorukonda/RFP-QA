
# streamlit_rfp_qa.py – memory‑friendly rewrite (June 2025)
"""
Streamlit app for Q&A over multiple RFP documents (PDF, Word, Excel) with a low
RAM footprint.

Key points
----------
* Streams chunks directly into FAISS – no huge lists kept in RAM.
* Stores only the first 400 characters of each chunk for answers.
* Uses the smaller paraphrase‑MiniLM‑L3‑v2 (256‑d) model.
* Skips files > 25 MB and Excel sheets > 25 000 rows.
* Everything stays in memory for the session; nothing is written to disk.
"""

from __future__ import annotations
import os
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")  # disable buggy watcher

from io import BytesIO
from pathlib import Path
from typing import Generator

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

try:
    import docx  # python-docx
except ImportError:
    docx = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# ----------------------------------------------------------------------------- #
MAX_FILE_MB     = 25_000_000  # bytes (≈ 25 MB)
MAX_SHEET_ROWS  = 25_000

def reject_large(file) -> bool:
    if file.size > MAX_FILE_MB:
        st.warning(f"{file.name} is {(file.size/1024**2):.1f} MB – skipped.")
        return True
    return False

def pdf_to_text(buf: BytesIO) -> str:
    if PyPDF2 is None:
        st.error("Please install PyPDF2 for PDF support")
        return ""
    reader = PyPDF2.PdfReader(buf)
    return "\n".join(p.extract_text() or "" for p in reader.pages)

def docx_to_text(buf: BytesIO) -> str:
    if docx is None:
        st.error("Please install python-docx for Word support")
        return ""
    document = docx.Document(buf)
    return "\n".join(p.text for p in document.paragraphs)

def excel_to_text(buf: BytesIO) -> str:
    sheets = pd.read_excel(buf, sheet_name=None, dtype=str, header=None)
    parts = []
    for name, df in sheets.items():
        if df.shape[0] > MAX_SHEET_ROWS:
            st.warning(f"Sheet {name} > {MAX_SHEET_ROWS:,} rows – skipped.")
            continue
        txt = df.fillna("").astype(str).agg(" ".join, axis=1).str.cat(sep="\n")
        parts.append(f"--- Sheet: {name} ---\n{txt}")
    return "\n".join(parts)

def chunk_stream(text: str, size: int = 512, overlap: int = 50) -> Generator[str, None, None]:
    words = text.split()
    i = 0
    while i < len(words):
        yield " ".join(words[i:i+size])
        i = max(i + size - overlap, 0)

# ----------------------------------------------------------------------------- #
st.set_page_config(page_title="RFP Q&A (low‑RAM)", page_icon="🤖", layout="wide")
st.title("📑 RFP Document Q&A – memory‑friendly edition")
st.write("Upload RFPs (PDF, Word, Excel). I’ll embed them on the fly – nothing is stored on disk.")

files = st.file_uploader(
    "Upload one or more files",
    type=["pdf","docx","doc","xls","xlsx"],
    accept_multiple_files=True)

if "model" not in st.session_state:
    with st.spinner("Loading small embedding model…"):
        st.session_state.model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
        dim = st.session_state.model.get_sentence_embedding_dimension()
        st.session_state.index     = faiss.IndexFlatIP(dim)
        st.session_state.snippets  = []

model    : SentenceTransformer = st.session_state.model
index    : faiss.IndexFlatIP    = st.session_state.index
snippets : list[str]            = st.session_state.snippets

if st.button("🔍 Scan / Extend Index") and files:
    with st.spinner("Processing documents…"):
        added = 0
        for file in files:
            if reject_large(file):
                continue
            buf = BytesIO(file.read())
            ext = Path(file.name).suffix.lower()
            if ext == ".pdf":
                text = pdf_to_text(buf)
            elif ext in {".doc",".docx"}:
                text = docx_to_text(buf)
            elif ext in {".xls",".xlsx"}:
                text = excel_to_text(buf)
            else:
                st.warning(f"Unsupported file {file.name} – skipped.")
                continue
            if not text.strip():
                st.warning(f"{file.name}: no extractable text.")
                continue
            for chunk in chunk_stream(text):
                vec = model.encode([chunk], normalize_embeddings=True)
                index.add(vec)
                snippets.append(chunk[:400])
                added += 1
        st.success(f"Added {added} chunks – total {len(snippets)}")

if snippets:
    q = st.text_input("Ask a question:")
    if q:
        vq = model.encode([q], normalize_embeddings=True)
        D,I = index.search(vq,k=5)
        scores, idxs = D.flatten(), I.flatten()
        if scores.size == 0 or scores[0] < 0.3:
            st.info("🔎 No relevant info found.")
        else:
            hits = [snippets[i] for i,s in zip(idxs,scores) if s >= 0.3][:3]
            st.markdown("### Answer")
            st.write("\n\n".join(hits))
            with st.expander("Similarities"):
                for r,(s,i) in enumerate(zip(scores,idxs),1):
                    st.write(f"**{r} – {s:.2f}** → {snippets[i][:200]}…")
else:
    st.info("Upload documents and click *Scan / Extend Index* first.")
