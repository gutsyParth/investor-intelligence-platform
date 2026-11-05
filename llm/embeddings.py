from functools import lru_cache

from openai import OpenAI

from config.settings import get_settings


@lru_cache(maxsize=1)
def get_embedding_client() -> OpenAI:
    settings = get_settings()

    if not settings.azure_openai_endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT is not configured")

    api_key = (
        settings.azure_openai_api_key.get_secret_value()
        if settings.azure_openai_api_key
        else ""
    )

    if not api_key:
        raise RuntimeError("AZURE_OPENAI_API_KEY is not configured")

    base_url = f"{settings.azure_openai_endpoint.rstrip('/')}/openai/v1/"

    return OpenAI(
        base_url=base_url,
        api_key=api_key,
    )


def generate_embeddings(
    chunks: list[str],
    client: OpenAI | None = None,
) -> list[list[float]]:
    if not chunks:
        return []

    if any(not chunk.strip() for chunk in chunks):
        raise ValueError("Embedding input cannot contain empty chunks")

    settings = get_settings()

    embedding_client = client if client is not None else get_embedding_client()

    response = embedding_client.embeddings.create(
        model=settings.azure_openai_embedding_deployment,
        input=chunks,
    )

    ordered_embeddings = sorted(
        response.data,
        key=lambda item: item.index,
    )

    if len(ordered_embeddings) != len(chunks):
        raise RuntimeError("Embedding response count does not match input count")

    return [item.embedding for item in ordered_embeddings]
