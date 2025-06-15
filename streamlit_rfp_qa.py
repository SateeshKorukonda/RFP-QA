# streamlit_rfp_qa.py
"""Streamlit app for Q&A over multiple RFP documents (PDF, Word, Excel).

All data lives only in memory during the user session – nothing is saved
or transmitted externally.

Run with:
    streamlit run streamlit_rfp_qa.py
"""

import os
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

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

################################################################################
# Utility functions
################################################################################

def extract_text_from_pdf(file_obj: BytesIO) -> str:
    """Extract text from a PDF file-like object using PyPDF2."""

    if PyPDF2 is None:
        st.error("PyPDF2 is not installed. Please install it with `pip install PyPDF2`.")
        return ""

    reader = PyPDF2.PdfReader(file_obj)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)

def extract_text_from_docx(file_obj: BytesIO) -> str:
    """Extract text from a .docx Word file."""

    if docx is None:
        st.error("python-docx is not installed. Please install it with `pip install python-docx`.")
        return ""

    doc = docx.Document(file_obj)
    paras = [p.text for p in doc.paragraphs]
    return "\n".join(paras)

def extract_text_from_excel(file_obj: BytesIO) -> str:
    """Read all sheets in an Excel file and return their contents as text."""

    xls = pd.read_excel(file_obj, sheet_name=None, dtype=str, header=None)
    all_sheets_text = []
    for sheet_name, df in xls.items():
        text = df.fillna("").astype(str).agg(" ".join, axis=1).str.cat(sep="\n")
        all_sheets_text.append(f"--- Sheet: {sheet_name} ---\n{text}")
    return "\n".join(all_sheets_text)

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
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
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks

################################################################################
# Streamlit UI
################################################################################

st.set_page_config(page_title="RFP Document Q&A", page_icon="🤖", layout="wide")
st.title("📑 RFP Document Q&A Assistant")
st.write(
    "Upload your RFP documents (PDF, Word, Excel). Once processed, ask any question and I’ll answer using only the uploaded content. **No data is stored or sent externally** – everything remains in memory for this session only."
)

uploaded_files = st.file_uploader(
    "Upload one or more RFP files",
    type=["pdf", "docx", "doc", "xlsx", "xls"],
    accept_multiple_files=True,
    help="Supported formats: PDF (.pdf), Word (.docx), Excel (.xlsx/.xls)."

)

if "model" not in st.session_state:
    with st.spinner("Loading embeddings model – one-time cost…"):
        st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2")

if st.button("🔍 Scan Documents") and uploaded_files:
    with st.spinner("Extracting and indexing text…"):
        all_chunks = []
        for file in uploaded_files:
            file_bytes = BytesIO(file.read())
            ext = Path(file.name).suffix.lower()
            text = ""
            if ext == ".pdf":
                text = extract_text_from_pdf(file_bytes)
            elif ext in {".docx", ".doc"}:
                text = extract_text_from_docx(file_bytes)
            elif ext in {".xlsx", ".xls"}:
                text = extract_text_from_excel(file_bytes)
            else:
                st.warning(f"Unsupported file type: {file.name}")
                continue
            if not text.strip():
                st.warning(f"No text extracted from {file.name}. It may be scanned or encrypted.")
                continue
            all_chunks.extend(chunk_text(text))
        if not all_chunks:
            st.error("No text found in any of the uploaded documents.")
        else:
            index, stored_chunks = build_faiss_index(all_chunks, st.session_state.model)
            st.session_state.index = index
            st.session_state.chunks = stored_chunks
            st.success(f"Indexed {len(stored_chunks)} text chunks from {len(uploaded_files)} document(s). You can now ask questions below.")

if "index" in st.session_state:
    question = st.text_input("Ask a question about the uploaded RFPs:")
    if question:
        model = st.session_state.model
        q_emb = model.encode([question], normalize_embeddings=True)
        D, I = st.session_state.index.search(q_emb, k=5)
        scores = D.flatten()
        idxs = I.flatten()
        best_score = scores[0] if len(scores) else 0.0
        THRESHOLD = 0.3
        if best_score < THRESHOLD:
            st.info("🔎 I couldn’t find information that answers your question in the provided documents.")
        else:
            top_chunks = [st.session_state.chunks[i] for i in idxs if scores[list(idxs).index(i)] >= THRESHOLD]
            answer = "\n\n".join(top_chunks[:3])
            st.markdown("### Answer")
            st.write(answer)
            with st.expander("Show confidence and matched chunks"):
                for rank, (score, i) in enumerate(zip(scores, idxs), start=1):
                    st.write(f"**Rank {rank} • Similarity {score:.2f}**\n\n{st.session_state.chunks[i][:1000]}…")
else:
    st.info("Upload documents and click **Scan Documents** to build the internal index.")

################################################################################
# End of file
################################################################################