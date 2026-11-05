import hashlib
from collections.abc import Callable

from azure.search.documents import SearchClient

from ingestion.chunker import chunk_document
from ingestion.pdf_parser import extract_pdf_text
from search.indexer import (
    build_search_documents,
    upload_search_documents,
)


def ingest_pdf(
    pdf_bytes: bytes,
    filename: str,
    embed_chunks: Callable[[list[str]], list[list[float]]],
    search_client: SearchClient,
) -> dict:
    if not pdf_bytes:
        raise ValueError("PDF file is empty")

    if not filename.strip():
        raise ValueError("Filename is required")

    # Step 1: Extract text from PDF
    page_count, extracted_text = extract_pdf_text(pdf_bytes)

    if not extracted_text.strip():
        raise ValueError("PDF contains no extractable text")

    # Step 2: Split extracted text into chunks
    chunks = chunk_document(extracted_text)

    if not chunks:
        raise ValueError("No text chunks generated")

    # Step 3: Generate embeddings in a batch
    embeddings = embed_chunks(chunks)

    # Step 4: Generate deterministic document identifier
    document_id = hashlib.sha256(pdf_bytes).hexdigest()

    # Step 5: Construct Azure AI Search documents
    documents = build_search_documents(
        document_id=document_id,
        filename=filename,
        chunks=chunks,
        embeddings=embeddings,
    )

    # Step 6: Upload documents to Azure AI Search
    uploaded_count = upload_search_documents(
        client=search_client,
        documents=documents,
    )

    # Step 7: Return ingestion result
    return {
        "document_id": document_id,
        "filename": filename,
        "page_count": page_count,
        "chunk_count": len(chunks),
        "indexed_count": uploaded_count,
    }
