from functools import lru_cache

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceExistsError
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex

from config.settings import get_settings
from search.index_schema import build_document_index


@lru_cache(maxsize=1)
def get_index_client() -> SearchIndexClient:
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

    return SearchIndexClient(
        endpoint=settings.azure_search_endpoint,
        credential=AzureKeyCredential(api_key),
    )


def ensure_document_index(
    client: SearchIndexClient | None = None,
) -> SearchIndex:
    index_client = client if client is not None else get_index_client()

    index_definition = build_document_index()

    try:
        return index_client.create_index(index_definition)

    except ResourceExistsError:
        existing_index = index_client.get_index(index_definition.name)

        existing_fields = {field.name: field for field in existing_index.fields}

        for expected_field in index_definition.fields:
            actual_field = existing_fields.get(expected_field.name)

            if actual_field is None:
                raise RuntimeError(
                    f"Existing index is missing field: {expected_field.name}"
                )

            if actual_field.type != expected_field.type:
                raise RuntimeError(f"Field type mismatch: {expected_field.name}")

            if bool(actual_field.key) != bool(expected_field.key):
                raise RuntimeError(
                    f"Field key configuration mismatch: {expected_field.name}"
                )

        expected_vector = next(
            field for field in index_definition.fields if field.name == "content_vector"
        )

        actual_vector = existing_fields["content_vector"]

        if (
            actual_vector.vector_search_dimensions
            != expected_vector.vector_search_dimensions
        ):
            raise RuntimeError("Existing index has incompatible vector dimensions")

        if (
            actual_vector.vector_search_profile_name
            != expected_vector.vector_search_profile_name
        ):
            raise RuntimeError("Existing index has incompatible vector profile")

        return existing_index


if __name__ == "__main__":
    index = ensure_document_index()

    print(f"Azure AI Search index ready: {index.name}")
