from functools import lru_cache

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from config.settings import get_settings


@lru_cache(maxsize=1)
def get_search_client() -> SearchClient:
    settings = get_settings()

    if not settings.azure_search_endpoint:
        raise RuntimeError("AZURE_SEARCH_ENDPOINT is not configured")

    api_key = (
        settings.azure_search_api_key.get_secret_value()
        if settings.azure_search_api_key
        else ""
    )

    if not api_key:
        raise RuntimeError("AZURE_SEARCH_API_KEY is not configured")

    return SearchClient(
        endpoint=settings.azure_search_endpoint,
        index_name=settings.azure_search_index_name,
        credential=AzureKeyCredential(api_key),
    )
