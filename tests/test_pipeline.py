from unittest.mock import Mock, patch

import pytest
from azure.search.documents import SearchClient

from ingestion.pipeline import ingest_pdf
from search.index_schema import EMBEDDING_DIMENSIONS


@patch("ingestion.pipeline.upload_search_documents")
@patch("ingestion.pipeline.chunk_document")
@patch("ingestion.pipeline.extract_pdf_text")
def test_ingest_pdf_success(
    mock_extract,
    mock_chunk,
    mock_upload,
):
    # Arrange
    mock_extract.return_value = (
        2,
        "Revenue increased. Profit increased.",
    )

    mock_chunk.return_value = [
        "Revenue increased.",
        "Profit increased.",
    ]

    mock_upload.return_value = 2

    mock_embed = Mock(return_value=[0.1] * EMBEDDING_DIMENSIONS)

    mock_search_client = Mock(spec=SearchClient)

    # Act
    result = ingest_pdf(
        pdf_bytes=b"%PDF-test-data",
        filename="annual_report.pdf",
        embed_text=mock_embed,
        search_client=mock_search_client,
    )

    # Assert
    assert result["filename"] == "annual_report.pdf"
    assert result["page_count"] == 2
    assert result["chunk_count"] == 2
    assert result["indexed_count"] == 2

    assert len(result["document_id"]) == 64

    assert mock_embed.call_count == 2
    mock_upload.assert_called_once()

    uploaded_documents = mock_upload.call_args.kwargs["documents"]

    assert len(uploaded_documents) == 2
    assert uploaded_documents[0]["content"] == "Revenue increased."
    assert len(uploaded_documents[0]["content_vector"]) == 1536


@patch("ingestion.pipeline.extract_pdf_text")
def test_ingest_pdf_empty_text(mock_extract):
    mock_extract.return_value = (1, "")

    with pytest.raises(
        ValueError,
        match="PDF contains no extractable text",
    ):
        ingest_pdf(
            pdf_bytes=b"%PDF-test-data",
            filename="empty.pdf",
            embed_text=Mock(),
            search_client=Mock(spec=SearchClient),
        )


def test_ingest_pdf_empty_file():
    with pytest.raises(
        ValueError,
        match="PDF file is empty",
    ):
        ingest_pdf(
            pdf_bytes=b"",
            filename="empty.pdf",
            embed_text=Mock(),
            search_client=Mock(spec=SearchClient),
        )


@patch("ingestion.pipeline.chunk_document")
@patch("ingestion.pipeline.extract_pdf_text")
def test_ingest_pdf_no_chunks(
    mock_extract,
    mock_chunk,
):
    mock_extract.return_value = (
        1,
        "Some extracted text",
    )

    mock_chunk.return_value = []

    with pytest.raises(
        ValueError,
        match="No text chunks generated",
    ):
        ingest_pdf(
            pdf_bytes=b"%PDF-test-data",
            filename="report.pdf",
            embed_text=Mock(),
            search_client=Mock(spec=SearchClient),
        )
