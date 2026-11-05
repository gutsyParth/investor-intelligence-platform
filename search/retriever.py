from collections.abc import Callable, Sequence

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from search.index_schema import EMBEDDING_DIMENSIONS

DEFAULT_TOP_K = 5
MAX_TOP_K = 20

RESULT_FIELDS = [
    "id",
    "document_id",
    "filename",
    "chunk_index",
    "content",
]


def retrieve_chunks(
    query: str,
    embed_query: Callable[[str], Sequence[float]],
    search_client: SearchClient,
    top_k: int = DEFAULT_TOP_K,
    hybrid: bool = True,
) -> list[dict]:
    """
    Retrieve relevant financial document chunks using
    vector similarity search or hybrid search.

    Args:
        query: User's natural-language question.
        embed_query: Function that converts text to an embedding.
        search_client: Azure AI Search client.
        top_k: Maximum number of results.
        hybrid: Whether to combine keyword and vector search.

    Returns:
        Ranked document chunks with metadata and search scores.
    """

    if not query.strip():
        raise ValueError("Search query cannot be empty")

    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")

    # Step 1: Generate query embedding
    query_vector = list(embed_query(query))

    if len(query_vector) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSIONS} embedding dimensions, "
            f"received {len(query_vector)}"
        )

    # Step 2: Construct vector query
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="content_vector",
        kind="vector",
    )

    # Step 3: Configure search request
    search_kwargs = {
        "vector_queries": [vector_query],
        "select": RESULT_FIELDS,
        "top": top_k,
    }

    # Hybrid search combines keyword matching and vector similarity
    if hybrid:
        search_kwargs["search_text"] = query

    # Step 4: Execute Azure AI Search query
    results = search_client.search(**search_kwargs)

    # Step 5: Transform results
    retrieved_chunks = []

    for result in results:
        retrieved_chunks.append(
            {
                "id": result["id"],
                "document_id": result["document_id"],
                "filename": result["filename"],
                "chunk_index": result["chunk_index"],
                "content": result["content"],
                "score": result.get("@search.score"),
            }
        )

    return retrieved_chunks
