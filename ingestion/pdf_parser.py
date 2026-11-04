from io import BytesIO

from pypdf import PdfReader


def extract_pdf_text(pdf_bytes: bytes) -> tuple[int, str]:
    reader = PdfReader(BytesIO(pdf_bytes))

    page_texts = []

    for page in reader.pages:
        text = page.extract_text() or ""
        page_texts.append(text)

    extracted_text = "\n\n".join(page_texts).strip()

    return len(reader.pages), extracted_text
