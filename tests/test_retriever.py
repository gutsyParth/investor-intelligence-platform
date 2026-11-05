from unittest.mock import Mock

import pytest
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from search.index_schema import EMBEDDING_DIMENSIONS
from search.retriever import (
    MAX_TOP_K,
    retrieve_chunks,
)


@pytest.fixture
def mock_search_client():
    client = Mock(spec=SearchClient)

    client.search.return_value = [
        {
            "id": "chunk-001",
            "document_id": "doc-123",
            "filename": "annual_report.pdf",
            "chunk_index": 0,
            "content": "Revenue increased by 15% in FY2025.",
            "@search.score": 0.95,
        },
        {
            "id": "chunk-002",
            "document_id": "doc-123",
            "filename": "annual_report.pdf",
            "chunk_index": 1,
            "content": "Operating profit increased by 12%.",
            "@search.score": 0.87,
        },
    ]

    return client


@pytest.fixture
def mock_embed_query():
    return Mock(return_value=[0.1] * EMBEDDING_DIMENSIONS)


def test_vector_search(
    mock_search_client,
    mock_embed_query,
):
    results = retrieve_chunks(
        query="What was revenue growth?",
        embed_query=mock_embed_query,
        search_client=mock_search_client,
        top_k=5,
        hybrid=False,
    )

    assert len(results) == 2

    assert results[0]["filename"] == "annual_report.pdf"
    assert results[0]["chunk_index"] == 0
    assert results[0]["score"] == 0.95

    mock_embed_query.assert_called_once_with("What was revenue growth?")

    mock_search_client.search.assert_called_once()

    search_kwargs = mock_search_client.search.call_args.kwargs

    assert "search_text" not in search_kwargs
    assert search_kwargs["top"] == 5

    vector_query = search_kwargs["vector_queries"][0]

    assert isinstance(vector_query, VectorizedQuery)
    assert vector_query.fields == "content_vector"
    assert vector_query.k_nearest_neighbors == 5

    assert len(vector_query.vector) == EMBEDDING_DIMENSIONS


def test_hybrid_search(
    mock_search_client,
    mock_embed_query,
):
    results = retrieve_chunks(
        query="EBITDA growth FY2025",
        embed_query=mock_embed_query,
        search_client=mock_search_client,
        top_k=3,
        hybrid=True,
    )

    assert len(results) == 2

    search_kwargs = mock_search_client.search.call_args.kwargs

    assert search_kwargs["search_text"] == "EBITDA growth FY2025"
    assert search_kwargs["top"] == 3
    assert len(search_kwargs["vector_queries"]) == 1

    assert "content" in search_kwargs["select"]
    assert "filename" in search_kwargs["select"]


def test_empty_query(
    mock_search_client,
    mock_embed_query,
):
    with pytest.raises(
        ValueError,
        match="Search query cannot be empty",
    ):
        retrieve_chunks(
            query="   ",
            embed_query=mock_embed_query,
            search_client=mock_search_client,
        )

    mock_embed_query.assert_not_called()
    mock_search_client.search.assert_not_called()


def test_invalid_embedding_dimensions(
    mock_search_client,
):
    invalid_embedding = Mock(return_value=[0.1, 0.2, 0.3])

    with pytest.raises(
        ValueError,
        match="Expected 1536 embedding dimensions",
    ):
        retrieve_chunks(
            query="Revenue growth",
            embed_query=invalid_embedding,
            search_client=mock_search_client,
        )

    mock_search_client.search.assert_not_called()


def test_invalid_top_k(
    mock_search_client,
    mock_embed_query,
):
    with pytest.raises(
        ValueError,
        match="top_k must be between",
    ):
        retrieve_chunks(
            query="Revenue growth",
            embed_query=mock_embed_query,
            search_client=mock_search_client,
            top_k=MAX_TOP_K + 1,
        )

    mock_search_client.search.assert_not_called()


def test_no_search_results(
    mock_search_client,
    mock_embed_query,
):
    mock_search_client.search.return_value = []

    results = retrieve_chunks(
        query="Unknown financial information",
        embed_query=mock_embed_query,
        search_client=mock_search_client,
    )

    assert results == []
