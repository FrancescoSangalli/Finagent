"""RAG retrieval from ChromaDB using similarity search."""

from typing import List

from src.rag.embeddings import get_embedding_function
from src.rag.vectorstore import get_collection


def retrieve(query: str, collection_name: str, k: int = 5) -> List[str]:
    """Retrieve top-k relevant chunks from a ChromaDB collection for a query."""
    try:
        embedder = get_embedding_function()
        query_embedding = embedder.embed_query(query)

        collection = get_collection(collection_name)
        count = collection.count()
        if count == 0:
            return []

        actual_k = min(k, count)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
        )
        documents = results.get("documents", [[]])[0]
        return documents

    except Exception:
        return []
