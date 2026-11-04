from fastapi import APIRouter, File, HTTPException, UploadFile
from pypdf.errors import PdfReadError

from ingestion.pdf_parser import extract_pdf_text

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/extract")
def extract_document(file: UploadFile = File(...)):
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

    try:
        page_count, extracted_text = extract_pdf_text(pdf_bytes)

    except (PdfReadError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Unable to extract text from PDF",
        ) from None

    if not extracted_text:
        raise HTTPException(
            status_code=422,
            detail="PDF contains no extractable text",
        )

    return {
        "filename": filename,
        "page_count": page_count,
        "character_count": len(extracted_text),
        "preview": extracted_text[:1000],
    }
