# FYP_NADRA_Interactive_Bot
A Retrieval-Augmented Generation System for NADRA Services
Final Year Project – Abrar Aqeel and Affan ur Rehman Khan

------------------------------------------------------------

1. Overview
The NADRA RAG Chatbot is an information assistant designed to answer questions about NADRA identity services using Retrieval-Augmented Generation (RAG). 
It uses a bilingual dataset (English + Urdu), embedding-based retrieval, and CPU-friendly language models to generate grounded, accurate responses.

------------------------------------------------------------

2. Key Features
- RAG pipeline with top-K retrieval (default K=3)
- English + Urdu support for queries and answers
- CPU-friendly answer generation model: google/flan-t5-base
- GPU-powered embedding ingestion (one time only)
- ChromaDB as vector database (local persistent mode)
- Streamlit front-end with clean UI
- Debug mode showing retrieved chunks
- Fully modular code aligned with project documentation

------------------------------------------------------------

3. Project Structure

    data_ingestion.py                 # Embedding + ChromaDB builder
    app.py                            # Streamlit RAG chatbot
    nadra_dataset_translated.json     # Final bilingual dataset
    chroma_db/                        # Auto-created database
    README.txt                        # Documentation

------------------------------------------------------------

4. Workflow

Step 1 — Dataset
You must have:
nadra_dataset_translated.json
This file contains aligned English and Urdu text for each chunk.

Step 2 — Embeddings (GPU recommended)
Run once:

    python data_ingestion.py

This will:
- Create nadra_english and nadra_urdu collections
- Store embeddings inside ./chroma_db
- Validate with sample queries

Step 3 — Chatbot
Run:

    streamlit run app.py

Open:
http://localhost:8501

Features:
- Ask questions in English or Urdu
- Choose output language
- See single final answer
- View debug details (retrieved chunks)

------------------------------------------------------------

5. Requirements

Install dependencies:

    pip install -r requirements.txt

Recommended Python version: 3.10+

------------------------------------------------------------

6. Deployment Options

- Local Streamlit app
- AWS EC2 (recommended)
- Azure VM
- Google Cloud VM

Inference runs fully on CPU.
Only embeddings require GPU (optional).

------------------------------------------------------------

7. Future Enhancements
- Voice input + voice output
- MetaHuman / avatar front-end
- Human-reviewed Urdu translations
- Admin interface for dataset updates
- Multi-user session support

------------------------------------------------------------

8. Author
Abrar Aqeel
Affan ur Rehman Khan
NADRA RAG Chatbot – Final Year Project

------------------------------------------------------------
