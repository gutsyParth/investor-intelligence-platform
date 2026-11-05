from collections.abc import Callable
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)

from ingestion.pipeline import ingest_pdf
from llm.embeddings import (
    generate_embeddings,
    get_embedding_client,
)
from search.client import get_search_client

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

MAX_FILE_SIZE = 10 * 1024 * 1024

DocumentIngestor = Callable[[bytes, str], dict]


def get_document_ingestor() -> DocumentIngestor:
    try:
        embedding_client = get_embedding_client()
        search_client = get_search_client()

    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document ingestion service is not configured",
        ) from exc

    def run_ingestion(
        pdf_bytes: bytes,
        filename: str,
    ) -> dict:
        return ingest_pdf(
            pdf_bytes=pdf_bytes,
            filename=filename,
            embed_chunks=lambda chunks: generate_embeddings(
                chunks,
                client=embedding_client,
            ),
            search_client=search_client,
        )

    return run_ingestion


@router.post(
    "/ingest",
    status_code=status.HTTP_201_CREATED,
)
def ingest_document(
    file: Annotated[UploadFile, File()],
    ingestor: Annotated[
        DocumentIngestor,
        Depends(get_document_ingestor),
    ],
):
    filename = file.filename or ""

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported",
        )

    pdf_bytes = file.file.read(MAX_FILE_SIZE + 1)

    if len(pdf_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="PDF exceeds the 10 MB limit",
        )

    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file",
        )

    return ingestor(pdf_bytes, filename)
