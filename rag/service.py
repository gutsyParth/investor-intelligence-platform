from collections.abc import Callable, Sequence

from azure.search.documents import SearchClient
from openai import OpenAI

from llm.generation import generate_grounded_answer
from search.retriever import retrieve_chunks

NO_CONTEXT_MESSAGE = (
    "I couldn't find relevant information in the "
    "uploaded documents to answer this question."
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
    # Step 1: Retrieve relevant document chunks
    retrieved_chunks = retrieve_chunks(
        query=question,
        embed_query=embed_query,
        search_client=search_client,
        top_k=top_k,
        hybrid=hybrid,
    )

    # Step 2: Handle missing context
    if not retrieved_chunks:
        return {
            "question": question,
            "answer": NO_CONTEXT_MESSAGE,
            "sources": [],
        }

    # Step 3: Generate an answer grounded in the retrieved chunks
    answer = generate_grounded_answer(
        question=question,
        chunks=retrieved_chunks,
        client=llm_client,
        model=chat_model,
    )

    # Step 4: Build source references
    sources = []

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        sources.append(
            {
                "reference": f"[{index}]",
                "document_id": chunk["document_id"],
                "filename": chunk["filename"],
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
            }
        )

    # Step 5: Return answer with supporting material
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }
