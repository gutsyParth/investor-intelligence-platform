from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from main import app
from routes.questions import get_question_answerer

client = TestClient(app)


@pytest.fixture
def mock_answerer():
    answerer = Mock(
        return_value={
            "question": "What was revenue growth?",
            "answer": ("Revenue increased by 15% in FY2025 [1]."),
            "sources": [
                {
                    "reference": "[1]",
                    "document_id": "doc-123",
                    "filename": "annual_report.pdf",
                    "chunk_index": 0,
                    "content": ("Revenue increased by 15% in FY2025."),
                }
            ],
        }
    )

    app.dependency_overrides[get_question_answerer] = lambda: answerer

    try:
        yield answerer
    finally:
        app.dependency_overrides.pop(
            get_question_answerer,
            None,
        )


def test_ask_question_success(mock_answerer):
    response = client.post(
        "/questions/ask",
        json={
            "question": "What was revenue growth?",
            "top_k": 5,
            "hybrid": True,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["question"] == "What was revenue growth?"
    assert "15%" in body["answer"]

    assert len(body["sources"]) == 1
    assert body["sources"][0]["reference"] == "[1]"

    mock_answerer.assert_called_once_with(
        "What was revenue growth?",
        5,
        True,
    )


def test_ask_question_empty_question(mock_answerer):
    response = client.post(
        "/questions/ask",
        json={
            "question": "",
        },
    )

    assert response.status_code == 422
    mock_answerer.assert_not_called()


def test_ask_question_whitespace_only(mock_answerer):
    response = client.post(
        "/questions/ask",
        json={
            "question": "   ",
        },
    )

    assert response.status_code == 422
    mock_answerer.assert_not_called()


def test_ask_question_invalid_top_k(mock_answerer):
    response = client.post(
        "/questions/ask",
        json={
            "question": "Revenue growth",
            "top_k": 100,
        },
    )

    assert response.status_code == 422
    mock_answerer.assert_not_called()


def test_ask_question_default_parameters(mock_answerer):
    response = client.post(
        "/questions/ask",
        json={
            "question": "What was revenue growth?",
        },
    )

    assert response.status_code == 200

    mock_answerer.assert_called_once_with(
        "What was revenue growth?",
        5,
        True,
    )


def test_question_answerer_dependency_wiring():
    with (
        patch("routes.questions.get_embedding_client") as mock_openai_client,
        patch("routes.questions.get_search_client") as mock_search_client,
        patch("routes.questions.generate_embeddings") as mock_generate,
        patch("routes.questions.answer_question") as mock_rag,
    ):
        mock_generate.return_value = [[0.1] * 1536]

        answerer = get_question_answerer()

        answerer(
            "What was revenue growth?",
            5,
            True,
        )

        mock_rag.assert_called_once()

        arguments = mock_rag.call_args.kwargs

        assert arguments["question"] == "What was revenue growth?"
        assert arguments["top_k"] == 5
        assert arguments["hybrid"] is True

        assert arguments["search_client"] is mock_search_client.return_value

        assert arguments["llm_client"] is mock_openai_client.return_value

        embedding_function = arguments["embed_query"]

        result = embedding_function("Revenue growth")

        assert result == [0.1] * 1536

        mock_generate.assert_called_once_with(
            ["Revenue growth"],
            client=mock_openai_client.return_value,
        )
