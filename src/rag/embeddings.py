"""Embedding function setup using sentence-transformers."""

from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedding_function() -> HuggingFaceEmbeddings:
    """Return a HuggingFace embedding function using all-MiniLM-L6-v2."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
