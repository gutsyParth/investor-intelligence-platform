from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from main import app
from routes.ingestion import get_document_ingestor

client = TestClient(app)


@pytest.fixture
def mock_ingestor():
    ingestor = Mock(
        return_value={
            "document_id": "abc123",
            "filename": "report.pdf",
            "page_count": 2,
            "chunk_count": 4,
            "indexed_count": 4,
        }
    )

    app.dependency_overrides[get_document_ingestor] = lambda: ingestor

    try:
        yield ingestor
    finally:
        app.dependency_overrides.pop(
            get_document_ingestor,
            None,
        )


def test_ingest_document_success(mock_ingestor):
    response = client.post(
        "/documents/ingest",
        files={
            "file": (
                "report.pdf",
                b"%PDF-test-data",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 201
    assert response.json()["indexed_count"] == 4

    mock_ingestor.assert_called_once_with(
        b"%PDF-test-data",
        "report.pdf",
    )


def test_ingest_document_invalid_extension(mock_ingestor):
    response = client.post(
        "/documents/ingest",
        files={
            "file": (
                "report.txt",
                b"Not a PDF",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    mock_ingestor.assert_not_called()


def test_ingest_document_invalid_pdf(mock_ingestor):
    response = client.post(
        "/documents/ingest",
        files={
            "file": (
                "report.pdf",
                b"Invalid content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    mock_ingestor.assert_not_called()


def test_ingest_document_too_large(mock_ingestor):
    response = client.post(
        "/documents/ingest",
        files={
            "file": (
                "large.pdf",
                b"%PDF-" + b"x" * (10 * 1024 * 1024),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 413
    mock_ingestor.assert_not_called()
