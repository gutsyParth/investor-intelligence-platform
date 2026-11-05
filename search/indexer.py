import hashlib
from collections.abc import Sequence

from azure.search.documents import SearchClient

from search.index_schema import EMBEDDING_DIMENSIONS


def build_search_documents(
    document_id: str,
    filename: str,
    chunks: Sequence[str],
    embeddings: Sequence[Sequence[float]],
) -> list[dict]:
    if not document_id or not filename:
        raise ValueError("Document ID and filename are required")

    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings count must match")

    documents = []

    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
        if not chunk.strip():
            raise ValueError(f"Chunk {index} is empty")

        if len(embedding) != EMBEDDING_DIMENSIONS:
            raise ValueError(f"Invalid embedding dimension for chunk {index}")

        chunk_id = hashlib.sha256(f"{document_id}:{index}".encode()).hexdigest()

        documents.append(
            {
                "id": chunk_id,
                "document_id": document_id,
                "filename": filename,
                "chunk_index": index,
                "content": chunk,
                "content_vector": list(embedding),
            }
        )

    return documents


def upload_search_documents(
    client: SearchClient,
    documents: list[dict],
) -> int:
    if not documents:
        return 0

    results = client.upload_documents(documents=documents)

    failed = [result.key for result in results if not result.succeeded]

    if failed or len(results) != len(documents):
        raise RuntimeError(f"Failed to upload all documents. Failed IDs: {failed}")

    return len(results)
