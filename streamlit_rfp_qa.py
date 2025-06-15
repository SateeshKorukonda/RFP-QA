
# streamlit_rfp_qa.py
"""
Streamlit app for Q&A over multiple RFP documents (PDF, Word, Excel).

* **Fix applied 2025-06-16**: disables Streamlit's file-watcher to avoid the
  `torch.classes.__path__` crash that affects Streamlit ≥ 1.42 when PyTorch is
  installed.  The workaround is a one-liner (`os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"`) placed **before** any `import streamlit`.

All data lives only in memory during the user session – nothing is saved
or transmitted externally.

Run with:
    streamlit run streamlit_rfp_qa.py
"""

from __future__ import annotations

###############################################################################
# Disable Streamlit's buggy file-watcher *before* importing streamlit
###############################################################################
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"  # ← NEW LINE (fix)

###############################################################################
# Standard libraries
###############################################################################
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

###############################################################################
# Third-party libraries
###############################################################################
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# Optional imports (guarded so the app still loads without them until needed)
try:
    import docx  # python-docx
except ImportError:
    docx = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

################################################################################
# --------------------------- Utility functions --------------------------------
################################################################################

def extract_text_from_pdf(file_obj: BytesIO) -> str:
    """Extract text from a PDF file-like object using PyPDF2."""
    if PyPDF2 is None:
        st.error("PyPDF2 is not installed. Please install it with `pip install PyPDF2`.")
        return ""
    reader = PyPDF2.PdfReader(file_obj)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_docx(file_obj: BytesIO) -> str:
    """Extract text from a .docx Word file."""
    if docx is None:
        st.error("python-docx is not installed. Please install it with `pip install python-docx`.")
        return ""
    document = docx.Document(file_obj)
    return "\n".join(p.text for p in document.paragraphs)


def extract_text_from_excel(file_obj: BytesIO) -> str:
    """Read all sheets in an Excel file and return their contents as a single string."""
    xls = pd.read_excel(file_obj, sheet_name=None, dtype=str, header=None)
    sheet_text = []
    for sheet_name, df in xls.items():
        text = df.fillna("").astype(str).agg(" ".join, axis=1).str.cat(sep="\n")
        sheet_text.append(f"--- Sheet: {sheet_name} ---\n{text}")
    return "\n".join(sheet_text)


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Simple whitespace-based text chunking with overlap."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def build_faiss_index(chunks: List[str], model: SentenceTransformer) -> Tuple[faiss.IndexFlatIP, List[str]]:
    """Return an in-memory FAISS index of normalised sentence-transformer embeddings."""
    vectors = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(vectors.shape[1])  # inner-product == cosine when vectors are normalised
    index.add(vectors)
    return index, chunks

################################################################################
# --------------------------- Streamlit UI -------------------------------------
################################################################################

st.set_page_config(page_title="RFP Document Q&A", page_icon="🤖", layout="wide")
st.title("📑 RFP Document Q&A Assistant")
st.write(
    "Upload your RFP documents (PDF, Word, Excel). Once processed, ask any question and I’ll answer using only the uploaded content. **No data is stored or sent externally** – everything remains in memory for this session only."
)

# ------------------ File upload ------------------
uploaded_files = st.file_uploader(
    "Upload one or more RFP files",
    type=["pdf", "docx", "doc", "xlsx", "xls"],
    accept_multiple_files=True,
    help="Supported formats: PDF (.pdf), Word (.docx), Excel (.xlsx /.xls).",
)

# ------------------ Load embedding model once per session ------------------
if "model" not in st.session_state:
    with st.spinner("Loading embeddings model (first-time only)…"):
        st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------ Build index ------------------
if st.button("🔍 Scan Documents") and uploaded_files:
    with st.spinner("Extracting text and building index…"):
        all_chunks: List[str] = []
        for file in uploaded_files:
            ext = Path(file.name).suffix.lower()
            file_bytes = BytesIO(file.read())
            match ext:
                case ".pdf":
                    text = extract_text_from_pdf(file_bytes)
                case ".doc" | ".docx":
                    text = extract_text_from_docx(file_bytes)
                case ".xls" | ".xlsx":
                    text = extract_text_from_excel(file_bytes)
                case _:
                    st.warning(f"Unsupported file type: {file.name} – skipped.")
                    continue
            if not text.strip():
                st.warning(f"No text extracted from {file.name}. Possible scan or encryption.")
                continue
            all_chunks.extend(chunk_text(text))

        if not all_chunks:
            st.error("No usable text found in any uploaded file.")
        else:
            idx, stored_chunks = build_faiss_index(all_chunks, st.session_state.model)
            st.session_state.index = idx
            st.session_state.chunks = stored_chunks
            st.success(f"Indexed {len(stored_chunks)} chunks from {len(uploaded_files)} document(s). Ask your question below.")

# ------------------ Q&A ------------------
if "index" in st.session_state:
    question = st.text_input("Ask a question about the uploaded RFPs:")
    if question:
        model: SentenceTransformer = st.session_state.model
        q_vec = model.encode([question], normalize_embeddings=True)
        D, I = st.session_state.index.search(q_vec, k=5)
        scores, idxs = D.flatten(), I.flatten()
        THRESHOLD = 0.3
        if scores.size == 0 or scores[0] < THRESHOLD:
            st.info("🔎 I couldn’t find information that answers your question in the provided documents.")
        else:
            top_chunks = [st.session_state.chunks[i] for i, s in zip(idxs, scores) if s >= THRESHOLD][:3]
            st.markdown("### Answer")
            st.write("\n\n".join(top_chunks))
            with st.expander("Show similarity scores and chunks"):
                for rank, (score, i) in enumerate(zip(scores, idxs), start=1):
                    st.write(f"**Rank {rank} • Similarity {score:.2f}**\n\n{st.session_state.chunks[i][:1000]}…")
else:
    st.info("Upload documents and click **Scan Documents** to build the index.")

################################################################################
# -------------------------------- End ----------------------------------------
################################################################################
