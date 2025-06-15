# RFP Document Q&A Streamlit App

This app lets you upload multiple RFP documents (PDF, Word, Excel) and
ask questions whose answers are extracted from the uploaded files **only**.

## Quick‑start

```bash
# 1. Install dependencies (CPU‑only)
pip install -r requirements.txt

# 2. Run the Streamlit app
streamlit run streamlit_rfp_qa.py
```

## How it works
1. Drag‑and‑drop your documents.
2. Click **Scan Documents** to build an in‑memory vector index (FAISS).
3. Ask questions in natural language.

No data is saved or sent externally; everything stays in memory for the life
of the Streamlit session.
