from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from config.settings import get_settings
from llm.embeddings import generate_embeddings, get_embedding_client
from rag.service import answer_question
from search.client import get_search_client

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
        question = value.strip()

        if not question:
            raise ValueError("Question cannot be empty")

        return question


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


QuestionAnswerer = Callable[[str, int, bool], dict]


def get_question_answerer() -> QuestionAnswerer:
    settings = get_settings()

    try:
        openai_client = get_embedding_client()
        search_client = get_search_client()

    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Question-answering service is not configured",
        ) from exc

    def run_question_answering(
        question: str,
        top_k: int,
        hybrid: bool,
    ) -> dict:
        return answer_question(
            question=question,
            embed_query=lambda text: generate_embeddings(
                [text],
                client=openai_client,
            )[0],
            search_client=search_client,
            llm_client=openai_client,
            chat_model=settings.azure_openai_chat_deployment,
            top_k=top_k,
            hybrid=hybrid,
        )

    return run_question_answering


@router.post(
    "/ask",
    response_model=AskQuestionResponse,
)
def ask_question(
    request: AskQuestionRequest,
    answerer: Annotated[
        QuestionAnswerer,
        Depends(get_question_answerer),
    ],
):
    return answerer(
        request.question,
        request.top_k,
        request.hybrid,
    )
