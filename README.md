# RAG Project - Bajaj Finance PDF QA

This project is a simple Retrieval-Augmented Generation (RAG) system built using:

- Python
- FAISS
- Hugging Face
- Sentence Transformers
- PyPDF

The system reads a Bajaj Finance PDF document, creates text chunks, converts them into embeddings, stores them in a FAISS vector database, and answers user questions from the PDF content.

---

# Features

- Extract text from PDF
- Chunk large text into smaller parts
- Create embeddings using Hugging Face model
- Store embeddings in FAISS
- Retrieve relevant chunks
- Ask questions from PDF

---

# Tech Stack

- Python 3.10
- FAISS
- sentence-transformers
- transformers
- pypdf
- numpy

---

# Project Structure

```bash
RAG_project_bajaj_finance/
│
├── data/
│   └── bajaj_terms.pdf
│
├── read_pdf.py
├── ask_pdf.py
├── requirements.txt
├── README.md
├── .env
└── .gitignore


Create virtual environment:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Run Project
python read_pdf.py
python ask_pdf.py
