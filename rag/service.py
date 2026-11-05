import re
from collections.abc import Callable, Sequence

from azure.search.documents import SearchClient
from openai import OpenAI

from llm.generation import generate_grounded_answer
from search.retriever import retrieve_chunks

NO_CONTEXT_MESSAGE = (
    "I couldn't find relevant information in the indexed documents "
    "to answer this question."
)


def answer_question(
    question: str,
    embed_query: Callable[[str], Sequence[float]],
    search_client: SearchClient,
    llm_client: OpenAI,
    chat_model: str,
    top_k: int = 5,
    hybrid: bool = True,
) -> dict:
    """
    Retrieve relevant document chunks and generate a grounded answer.

    Every numbered citation in the answer must correspond to a
    retrieved source.
    """

    chunks = retrieve_chunks(
        query=question,
        embed_query=embed_query,
        search_client=search_client,
        top_k=top_k,
        hybrid=hybrid,
    )

    if not chunks:
        return {
            "question": question,
            "answer": NO_CONTEXT_MESSAGE,
            "sources": [],
        }

    answer = generate_grounded_answer(
        question=question,
        chunks=chunks,
        client=llm_client,
        model=chat_model,
    )

    # Extract references such as [1], [2], etc.
    citations = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}

    valid_citations = set(range(1, len(chunks) + 1))

    invalid_citations = citations - valid_citations

    if invalid_citations:
        raise RuntimeError("Generated answer contains invalid source references")

    sources = [
        {
            "reference": f"[{index}]",
            "document_id": chunk["document_id"],
            "filename": chunk["filename"],
            "chunk_index": chunk["chunk_index"],
            "content": chunk["content"],
        }
        for index, chunk in enumerate(chunks, start=1)
    ]

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }
