from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from llm.generation import generate_grounded_answer
from rag.service import NO_CONTEXT_MESSAGE, answer_question


@pytest.fixture
def sample_chunks():
    return [
        {
            "id": "chunk-001",
            "document_id": "doc-123",
            "filename": "annual_report.pdf",
            "chunk_index": 0,
            "content": "Revenue increased by 15% in FY2025.",
            "score": 0.95,
        },
        {
            "id": "chunk-002",
            "document_id": "doc-123",
            "filename": "annual_report.pdf",
            "chunk_index": 1,
            "content": "Operating profit increased by 12%.",
            "score": 0.87,
        },
    ]


def test_generate_grounded_answer(sample_chunks):
    mock_client = Mock()

    mock_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=("Revenue increased by 15% in FY2025 [1].")
                )
            )
        ]
    )

    answer = generate_grounded_answer(
        question="What was revenue growth?",
        chunks=sample_chunks,
        client=mock_client,
        model="test-chat-deployment",
    )

    assert "15%" in answer
    assert "[1]" in answer

    mock_client.chat.completions.create.assert_called_once()

    kwargs = mock_client.chat.completions.create.call_args.kwargs

    assert kwargs["model"] == "test-chat-deployment"
    assert kwargs["temperature"] == 0

    assert kwargs["messages"][0]["role"] == "system"
    assert kwargs["messages"][1]["role"] == "user"

    user_prompt = kwargs["messages"][1]["content"]

    assert "What was revenue growth?" in user_prompt
    assert "Revenue increased by 15%" in user_prompt
    assert "annual_report.pdf" in user_prompt


def test_generate_answer_empty_question(sample_chunks):
    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        generate_grounded_answer(
            question=" ",
            chunks=sample_chunks,
            client=Mock(),
            model="test-chat-deployment",
        )


def test_generate_answer_empty_context():
    with pytest.raises(
        ValueError,
        match="Retrieved chunks cannot be empty",
    ):
        generate_grounded_answer(
            question="What was revenue growth?",
            chunks=[],
            client=Mock(),
            model="test-chat-deployment",
        )


def test_generate_answer_empty_llm_response(sample_chunks):
    mock_client = Mock()

    mock_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=""))]
    )

    with pytest.raises(
        RuntimeError,
        match="LLM returned an empty answer",
    ):
        generate_grounded_answer(
            question="What was revenue growth?",
            chunks=sample_chunks,
            client=mock_client,
            model="test-chat-deployment",
        )


@patch("rag.service.generate_grounded_answer")
@patch("rag.service.retrieve_chunks")
def test_answer_question_success(
    mock_retrieve,
    mock_generate,
    sample_chunks,
):
    mock_retrieve.return_value = sample_chunks

    mock_generate.return_value = "Revenue increased by 15% in FY2025 [1]."

    mock_embed = Mock()
    mock_search_client = Mock()
    mock_llm_client = Mock()

    result = answer_question(
        question="What was revenue growth?",
        embed_query=mock_embed,
        search_client=mock_search_client,
        llm_client=mock_llm_client,
        chat_model="test-chat-deployment",
        top_k=5,
    )

    assert result["question"] == "What was revenue growth?"
    assert "15%" in result["answer"]

    assert len(result["sources"]) == 2

    assert result["sources"][0]["reference"] == "[1]"
    assert result["sources"][0]["filename"] == "annual_report.pdf"

    mock_retrieve.assert_called_once()
    mock_generate.assert_called_once()


@patch("rag.service.generate_grounded_answer")
@patch("rag.service.retrieve_chunks")
def test_answer_question_no_results(
    mock_retrieve,
    mock_generate,
):
    mock_retrieve.return_value = []

    result = answer_question(
        question="What was revenue growth?",
        embed_query=Mock(),
        search_client=Mock(),
        llm_client=Mock(),
        chat_model="test-chat-deployment",
    )

    assert result["answer"] == NO_CONTEXT_MESSAGE
    assert result["sources"] == []

    mock_generate.assert_not_called()
