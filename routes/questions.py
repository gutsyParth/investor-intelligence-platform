import logging
from collections.abc import Callable
from typing import Any

from azure.core.exceptions import AzureError
from fastapi import APIRouter, Depends, HTTPException
from openai import OpenAIError
from pydantic import BaseModel, Field, field_validator

from config.settings import get_settings
from llm.embeddings import generate_embeddings, get_embedding_client
from rag.service import answer_question
from search.client import get_search_client

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/questions",
    tags=["Questions"],
)


class AskQuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    hybrid: bool = True

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Question cannot be empty")

        return value


class SourceReference(BaseModel):
    reference: str
    document_id: str
    filename: str
    chunk_index: int
    content: str


class AskQuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceReference]


def get_question_answerer() -> Callable[..., dict[str, Any]]:
    """
    Build the RAG service using configured Azure clients.
    """

    settings = get_settings()

    try:
        openai_client = get_embedding_client()
        search_client = get_search_client()

    except RuntimeError as exc:
        logger.warning(
            "RAG service configuration is incomplete: %s",
            type(exc).__name__,
        )

        raise HTTPException(
            status_code=503,
            detail="AI services are not configured",
        ) from exc

    def question_answerer(
        question: str,
        top_k: int,
        hybrid: bool,
    ) -> dict[str, Any]:

        def embed_query(text: str) -> list[float]:
            embeddings = generate_embeddings(
                chunks=[text],
                client=openai_client,
            )

            return embeddings[0]

        return answer_question(
            question=question,
            embed_query=embed_query,
            search_client=search_client,
            llm_client=openai_client,
            chat_model=settings.azure_openai_chat_deployment,
            top_k=top_k,
            hybrid=hybrid,
        )

    return question_answerer


@router.post(
    "/ask",
    response_model=AskQuestionResponse,
)
def ask_question(
    request: AskQuestionRequest,
    answerer: Callable[..., dict[str, Any]] = Depends(get_question_answerer),
) -> dict[str, Any]:
    """
    Answer a financial question using indexed documents.
    """

    try:
        return answerer(
            question=request.question,
            top_k=request.top_k,
            hybrid=request.hybrid,
        )

    except (AzureError, OpenAIError, RuntimeError, ValueError):
        logger.exception("RAG question-answering failed")

        raise HTTPException(
            status_code=502,
            detail=("The AI question-answering service is temporarily unavailable"),
        ) from None
