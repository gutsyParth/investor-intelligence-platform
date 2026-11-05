from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from azure.search.documents import SearchClient

from search.index_schema import EMBEDDING_DIMENSIONS
from search.indexer import (
    build_search_documents,
    upload_search_documents,
)


@pytest.fixture
def sample_embeddings():
    return [
        [0.1] * EMBEDDING_DIMENSIONS,
        [0.2] * EMBEDDING_DIMENSIONS,
    ]


def test_build_search_documents(sample_embeddings):
    chunks = [
        "Revenue increased by 15%.",
        "Operating profit increased by 10%.",
    ]

    documents = build_search_documents(
        document_id="doc-123",
        filename="annual_report.pdf",
        chunks=chunks,
        embeddings=sample_embeddings,
    )

    assert len(documents) == 2

    assert documents[0]["document_id"] == "doc-123"
    assert documents[0]["chunk_index"] == 0
    assert documents[0]["content"] == chunks[0]
    assert documents[0]["content_vector"] == sample_embeddings[0]

    assert documents[0]["id"] != documents[1]["id"]


def test_document_ids_are_deterministic(sample_embeddings):
    arguments = {
        "document_id": "doc-123",
        "filename": "annual_report.pdf",
        "chunks": ["First chunk", "Second chunk"],
        "embeddings": sample_embeddings,
    }

    first = build_search_documents(**arguments)
    second = build_search_documents(**arguments)

    assert [doc["id"] for doc in first] == [doc["id"] for doc in second]


def test_mismatched_chunks_and_embeddings():
    with pytest.raises(ValueError):
        build_search_documents(
            document_id="doc-123",
            filename="report.pdf",
            chunks=["Chunk one", "Chunk two"],
            embeddings=[[0.1] * EMBEDDING_DIMENSIONS],
        )


def test_invalid_embedding_dimensions():
    with pytest.raises(ValueError):
        build_search_documents(
            document_id="doc-123",
            filename="report.pdf",
            chunks=["Sample text"],
            embeddings=[[0.1, 0.2]],
        )


def test_upload_search_documents_success():
    client = Mock(spec=SearchClient)

    client.upload_documents.return_value = [
        SimpleNamespace(key="chunk-1", succeeded=True),
    ]

    documents = [{"id": "chunk-1"}]

    uploaded_count = upload_search_documents(
        client,
        documents,
    )

    assert uploaded_count == 1
    client.upload_documents.assert_called_once_with(documents=documents)


def test_upload_search_documents_failure():
    client = Mock(spec=SearchClient)

    client.upload_documents.return_value = [
        SimpleNamespace(key="chunk-1", succeeded=False),
    ]

    with pytest.raises(RuntimeError):
        upload_search_documents(
            client,
            [{"id": "chunk-1"}],
        )
