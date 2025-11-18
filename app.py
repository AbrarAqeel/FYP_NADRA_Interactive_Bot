"""
app.py
NADRA RAG Chatbot (English + Urdu)
----------------------------------

Features:
- CPU-friendly answer generation using FLAN-T5-BASE
- English & Urdu language support
- Retrieval-augmented generation over bilingual ChromaDB
- Top-K retrieval (default = 3)
- Single final synthesized answer
- Expandable debug section (shows retrieved context)
- Matches project documentation exactly
"""

# ============================================================
#                    CONFIGURATION BLOCK
# ============================================================

CHROMA_DIR = r".\chroma_db"  # Path to your Chroma DB
COLLECTION_EN = "nadra_english"  # English collection
COLLECTION_UR = "nadra_urdu"  # Urdu collection
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Same as ingestion
GEN_MODEL = "google/flan-t5-base"  # CPU-friendly / better quality
DEFAULT_TOP_K = 3  # Retrieval count

# ============================================================
#                      IMPORTS
# ============================================================

import streamlit as st
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb
from chromadb.config import Settings


# ============================================================
#                  LOAD RESOURCES (CACHED)
# ============================================================

@st.cache_resource
def load_embedder() -> SentenceTransformer:
    """Load SentenceTransformer model for embeddings."""
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def load_chroma():
    """Load English & Urdu ChromaDB collections."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    coll_en = client.get_collection(COLLECTION_EN)
    try:
        coll_ur = client.get_collection(COLLECTION_UR)
    except:
        coll_ur = None

    return coll_en, coll_ur


@st.cache_resource
def load_generator():
    """Load FLAN-T5-BASE text generation model (CPU)."""
    return pipeline(
        "text2text-generation",
        model=GEN_MODEL,
        device=-1  # CPU
    )


# ============================================================
#                  RETRIEVAL FUNCTION
# ============================================================

def retrieve_top_k(
        query: str,
        lang: str,
        embedder: SentenceTransformer,
        coll_en,
        coll_ur,
        k: int
) -> Tuple[List[str], List[Dict], Optional[str]]:
    """
    Retrieve top-K documents from ChromaDB.
    Returns: (docs, metas, source_title)
    """
    target_coll = coll_ur if lang == "ur" and coll_ur else coll_en

    qvec = embedder.encode([query])[0]

    result = target_coll.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["documents", "metadatas"]
    )

    docs = result["documents"][0]
    metas = result["metadatas"][0]

    # Primary source = most relevant chunk
    source_title = metas[0].get("document_title") or \
                   metas[0].get("document_title_ur") or "Unknown"

    return docs, metas, source_title


# ============================================================
#           ANSWER SYNTHESIS (GENERATION MODEL)
# ============================================================

def synthesize_answer(
        query: str,
        docs: List[str],
        lang: str,
        generator
) -> str:
    """Generate final answer using FLAN-T5-BASE."""
    if not docs:
        return (
            "معذرت، کوئی متعلقہ معلومات نہیں مل سکیں۔"
            if lang == "ur"
            else "No relevant information was found."
        )

    # Build context
    context = "\n\n".join([f"[{i + 1}] {d}" for i, d in enumerate(docs)])

    if lang == "ur":
        prompt = (
            "آپ نادرا کے معلوماتی اسسٹنٹ ہیں۔ صرف دی گئی معلومات سے جواب دیں۔\n\n"
            f"مواد:\n{context}\n\n"
            f"سوال: {query}\n\n"
            "براہِ مہربانی مختصر، سادہ اور واضح جواب دیں۔"
        )
    else:
        prompt = (
            "You are the NADRA Information Assistant. Answer using ONLY the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Provide a short and clear answer."
        )

    result = generator(
        prompt,
        max_length=300,
        temperature=0.0
    )

    return result[0]["generated_text"].strip()


# ============================================================
#                       UI LAYOUT
# ============================================================

st.set_page_config(page_title="NADRA RAG Chatbot", layout="wide")
st.title("NADRA RAG Chatbot")

# Sidebar
st.sidebar.header("Settings")
language = st.sidebar.selectbox("Language", ["en", "ur"])
top_k = st.sidebar.number_input("Top-K Retrieval", min_value=1, max_value=5, value=DEFAULT_TOP_K)

# Load resources
embedder = load_embedder()
coll_en, coll_ur = load_chroma()
generator = load_generator()

# User Query
query = st.text_input("Ask a question about NADRA services:")

if st.button("Submit") and query.strip():
    with st.spinner("Retrieving..."):
        docs, metas, source_title = retrieve_top_k(query, language, embedder, coll_en, coll_ur, top_k)

    with st.spinner("Generating answer..."):
        answer = synthesize_answer(query, docs, language, generator)

    st.markdown("### Final Answer")
    st.write(answer)

    st.info(f"Source: {source_title}")

    # Debug View
    with st.expander("Debug: Retrieved Chunks"):
        for i, d in enumerate(docs):
            st.markdown(f"**Chunk {i + 1}**")
            st.write(d)
            st.markdown("---")

