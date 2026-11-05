from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from main import app
from routes.questions import get_question_answerer
from search.index_schema import EMBEDDING_DIMENSIONS

QUESTION = "What was the company's revenue?"


@pytest.fixture
def mocked_cloud(monkeypatch):
    """
    Replace external Azure dependencies with in-memory mocks.

    The real application flow remains intact:
    FastAPI -> Embedding -> Search -> Generation -> Response.
    """

    openai_client = MagicMock()
    search_client = MagicMock()

    # Mock embedding model response.
    openai_client.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(
                index=0,
                embedding=[0.01] * EMBEDDING_DIMENSIONS,
            )
        ]
    )

    # Mock Azure AI Search response.
    search_client.search.return_value = [
        {
            "id": "chunk-001",
            "document_id": "document-001",
            "filename": "annual_report.pdf",
            "chunk_index": 0,
            "content": ("The company reported annual revenue of $12 million in 2025."),
            "@search.score": 0.95,
        },
        {
            "id": "chunk-002",
            "document_id": "document-001",
            "filename": "annual_report.pdf",
            "chunk_index": 1,
            "content": (
                "Revenue increased by 15 percent compared with the previous year."
            ),
            "@search.score": 0.89,
        },
    ]

    # Mock chat completion response.
    openai_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=("The company reported revenue of $12 million in 2025 [1].")
                )
            )
        ]
    )

    monkeypatch.setattr(
        "routes.questions.get_embedding_client",
        lambda: openai_client,
    )

    monkeypatch.setattr(
        "routes.questions.get_search_client",
        lambda: search_client,
    )

    # Preserve any dependency overrides from other tests.
    previous_overrides = app.dependency_overrides.copy()

    app.dependency_overrides.pop(
        get_question_answerer,
        None,
    )

    try:
        yield (
            TestClient(app),
            openai_client,
            search_client,
        )

    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous_overrides)


def test_complete_rag_flow(mocked_cloud):
    client, openai_client, search_client = mocked_cloud

    response = client.post(
        "/questions/ask",
        json={
            "question": QUESTION,
            "top_k": 5,
            "hybrid": True,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["question"] == QUESTION
    assert "$12 million" in body["answer"]
    assert "[1]" in body["answer"]

    assert len(body["sources"]) == 2

    assert body["sources"][0]["reference"] == "[1]"
    assert body["sources"][1]["reference"] == "[2]"

    assert body["sources"][0]["filename"] == "annual_report.pdf"

    # Embedding service was invoked.
    openai_client.embeddings.create.assert_called_once()

    embedding_args = openai_client.embeddings.create.call_args.kwargs

    assert embedding_args["input"] == [QUESTION]

    # Search service was invoked.
    search_client.search.assert_called_once()

    search_args = search_client.search.call_args.kwargs

    assert search_args["top"] == 5
    assert search_args["search_text"] == QUESTION
    assert len(search_args["vector_queries"]) == 1

    # Chat generation was invoked.
    openai_client.chat.completions.create.assert_called_once()


def test_no_relevant_documents(mocked_cloud):
    client, openai_client, search_client = mocked_cloud

    search_client.search.return_value = []

    response = client.post(
        "/questions/ask",
        json={"question": QUESTION},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["sources"] == []
    assert body["answer"]

    # LLM must not generate an answer without context.
    openai_client.chat.completions.create.assert_not_called()


def test_invalid_citation_is_rejected(mocked_cloud):
    client, openai_client, _ = mocked_cloud

    openai_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=("The company earned $12 million [99].")
                )
            )
        ]
    )

    response = client.post(
        "/questions/ask",
        json={"question": QUESTION},
    )

    assert response.status_code == 502


def test_embedding_failure_returns_502(mocked_cloud):
    client, openai_client, _ = mocked_cloud

    openai_client.embeddings.create.side_effect = RuntimeError(
        "private embedding failure"
    )

    response = client.post(
        "/questions/ask",
        json={"question": QUESTION},
    )

    assert response.status_code == 502

    # Internal errors must not leak into API responses.
    assert "private embedding failure" not in response.text


def test_search_failure_returns_502(mocked_cloud):
    client, _, search_client = mocked_cloud

    search_client.search.side_effect = RuntimeError("Azure Search connection failed")

    response = client.post(
        "/questions/ask",
        json={"question": QUESTION},
    )

    assert response.status_code == 502


def test_generation_failure_returns_502(mocked_cloud):
    client, openai_client, _ = mocked_cloud

    openai_client.chat.completions.create.side_effect = RuntimeError(
        "Generation failed"
    )

    response = client.post(
        "/questions/ask",
        json={"question": QUESTION},
    )

    assert response.status_code == 502


def test_invalid_question_returns_422(mocked_cloud):
    client, _, _ = mocked_cloud

    response = client.post(
        "/questions/ask",
        json={"question": "   "},
    )

    assert response.status_code == 422


def test_invalid_top_k_returns_422(mocked_cloud):
    client, _, _ = mocked_cloud

    response = client.post(
        "/questions/ask",
        json={
            "question": QUESTION,
            "top_k": 25,
        },
    )

    assert response.status_code == 422


def test_vector_only_search(mocked_cloud):
    client, _, search_client = mocked_cloud

    response = client.post(
        "/questions/ask",
        json={
            "question": QUESTION,
            "top_k": 3,
            "hybrid": False,
        },
    )

    assert response.status_code == 200

    search_args = search_client.search.call_args.kwargs

    assert search_args["top"] == 3
    assert "search_text" not in search_args


def test_missing_ai_configuration_returns_503(
    mocked_cloud,
    monkeypatch,
):
    client, _, _ = mocked_cloud

    def unavailable_client():
        raise RuntimeError("Azure API key missing")

    monkeypatch.setattr(
        "routes.questions.get_embedding_client",
        unavailable_client,
    )

    response = client.post(
        "/questions/ask",
        json={"question": QUESTION},
    )

    assert response.status_code == 503

    assert "Azure API key missing" not in response.text
