"""ChromaDB vectorstore operations with local persistence."""

import os
from typing import List

import chromadb
from chromadb.config import Settings

from src.rag.embeddings import get_embedding_function

CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "chromadb")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def _get_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client pointed at the local data directory."""
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def index_documents(docs: List[str], collection_name: str) -> None:
    """Chunk and index a list of document strings into a ChromaDB collection."""
    client = _get_client()
    embedder = get_embedding_function()

    collection = client.get_or_create_collection(name=collection_name)

    all_chunks = []
    for doc in docs:
        all_chunks.extend(_chunk_text(doc))

    if not all_chunks:
        return

    embeddings = embedder.embed_documents(all_chunks)
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]

    collection.upsert(
        documents=all_chunks,
        embeddings=embeddings,
        ids=ids,
    )


def get_collection(collection_name: str) -> chromadb.Collection:
    """Return an existing ChromaDB collection by name."""
    client = _get_client()
    return client.get_or_create_collection(name=collection_name)
