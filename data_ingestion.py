"""
data_ingestion.py
------------------

This script:
1. Loads your existing bilingual JSON file (English + Urdu).
2. Embeds English text into a Chroma collection: `nadra_english`.
3. Embeds Urdu text into a Chroma collection: `nadra_urdu`.
4. Uses GPU for embeddings (if available).
5. Performs validation:
    - Checks Chroma directory exists/created
    - Confirms both collections exist
    - Confirms embeddings count
    - Runs a test query and prints top-3 retrieved chunks

This script is modular, documented, and follows your project documentation.
"""

# ============================================================
#                     CONFIGURATION BLOCK
# ============================================================

JSON_PATH: str = r"C:\Users\PC\Desktop\AbrarAqeel\FYP\data\nadra_dataset_translated.json"  # Your final bilingual JSON
CHROMA_DIR: str = "./chroma_db"  # Chroma storage directory
COLLECTION_EN: str = "nadra_english"  # English collection name
COLLECTION_UR: str = "nadra_urdu"  # Urdu collection name
EMBED_MODEL: str = "sentence-transformers/all-mpnet-base-v2"  # GPU friendly
TEST_QUERY_EN: str = "CNIC renewal process"
TEST_QUERY_UR: str = "شناختی کارڈ کی تجدید کا طریقہ"

# ============================================================
#                     IMPORTS
# ============================================================

import json
import os
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm


# ============================================================
#              LOAD DATASET (Modular Function)
# ============================================================

def load_dataset(json_path: str) -> List[Dict]:
    """
    Load bilingual JSON dataset.

    Parameters:
        json_path (str): Path to the JSON file.

    Returns:
        List[Dict]: List of dataset chunks.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
#              INITIALIZE CHROMA (Modular Function)
# ============================================================

def init_chroma(chroma_dir: str) -> chromadb.Client:
    """
    Initialize ChromaDB client.

    Parameters:
        chroma_dir (str): Folder to store Chroma collections.

    Returns:
        chromadb.Client: Chroma client instance.
    """
    os.makedirs(chroma_dir, exist_ok=True)
    return chromadb.PersistentClient(path=chroma_dir)


# ============================================================
#              CREATE COLLECTION (Modular)
# ============================================================

def create_collection(client: chromadb.Client, name: str):
    """
    Create or load a Chroma collection.

    Parameters:
        client: Chroma client instance.
        name: Collection name.

    Returns:
        Chroma Collection object.
    """
    return client.get_or_create_collection(name=name)


# ============================================================
#              EMBEDDING FUNCTION (Modular)
# ============================================================

def embed_texts(model: SentenceTransformer, texts: List[str]):
    """
    Generate embeddings using GPU if available.

    Parameters:
        model: SentenceTransformer model instance.
        texts: List of text strings.

    Returns:
        List[List[float]]: Embedding vectors.
    """
    return model.encode(texts, batch_size=16, show_progress_bar=True)


# ============================================================
#              UPSERT INTO CHROMA (Modular)
# ============================================================

def upsert_embeddings(collection, ids: List[str], texts: List[str], metas: List[Dict], embeddings):
    """
    Upsert embeddings into a Chroma collection.

    Parameters:
        collection: Chroma collection instance.
        ids: Chunk IDs.
        texts: List of documents.
        metas: Metadata list.
        embeddings: Embedding vectors.
    """
    for i in tqdm(range(len(ids)), desc=f"Upserting into {collection.name}", unit="chunk"):
        collection.add(
            ids=[ids[i]],
            documents=[texts[i]],
            metadatas=[metas[i]],
            embeddings=[embeddings[i].tolist()]
        )


# ============================================================
#        VALIDATION: CHECK COLLECTION + SAMPLE QUERY
# ============================================================

def test_collection(collection, test_query: str, model: SentenceTransformer):
    """
    Run a sample query against a Chroma collection.

    Parameters:
        collection: Chroma collection instance.
        test_query: Query to test retrieval.
        model: Embedding model.

    Prints:
        Top-3 retrieved chunks.
    """
    qvec = model.encode([test_query])[0]

    res = collection.query(
        query_embeddings=[qvec],
        n_results=3,
        include=["documents", "metadatas"]
    )

    print("\n--- TEST QUERY RESULTS ---")
    for i, doc in enumerate(res["documents"][0]):
        title = res["metadatas"][0][i].get("document_title", "N/A")
        print(f"\nResult {i + 1}:")
        print("Title:", title)
        print("Text:", doc[:300], "...")


# ============================================================
#                     MAIN WORKFLOW
# ============================================================

def main():
    print("Loading dataset...")
    dataset = load_dataset(JSON_PATH)
    print("Total chunks loaded:", len(dataset))

    print("\nLoading embedding model on GPU if available...")
    model = SentenceTransformer(EMBED_MODEL)

    print("\nInitializing Chroma DB...")
    client = init_chroma(CHROMA_DIR)

    # ---- Prepare English data ----
    print("\nPreparing English data...")
    ids_en = []
    texts_en = []
    metas_en = []
    for x in tqdm(dataset, desc="Preparing EN", unit="chunk"):
        ids_en.append(x["id"])
        texts_en.append(x["text"])
        metas_en.append({"document_title": x["document_title"]})

    # ---- Prepare Urdu data ----
    print("\nPreparing Urdu data...")
    ids_ur = []
    texts_ur = []
    metas_ur = []
    for x in tqdm(dataset, desc="Preparing UR", unit="chunk"):
        ids_ur.append(x["id"])
        texts_ur.append(x["text_ur"] or "")
        metas_ur.append({"document_title_ur": x["document_title_ur"]})

    # ---- EMBED ENGLISH ----
    print("\nEmbedding English texts...")
    emb_en = embed_texts(model, texts_en)

    coll_en = create_collection(client, COLLECTION_EN)
    upsert_embeddings(coll_en, ids_en, texts_en, metas_en, emb_en)
    print("English embeddings inserted.")

    # ---- EMBED URDU ----
    print("\nEmbedding Urdu texts...")
    emb_ur = embed_texts(model, texts_ur)

    coll_ur = create_collection(client, COLLECTION_UR)
    upsert_embeddings(coll_ur, ids_ur, texts_ur, metas_ur, emb_ur)
    print("Urdu embeddings inserted.")

    # ---- VALIDATION ----
    print("\nValidating collections...")
    print("English collection count:", coll_en.count())
    print("Urdu collection count:", coll_ur.count())

    test_collection(coll_en, TEST_QUERY_EN, model)
    test_collection(coll_ur, TEST_QUERY_UR, model)

    print("\n=== INGESTION COMPLETE ===")
    print("Chroma DB stored at:", CHROMA_DIR)


if __name__ == "__main__":
    main()
