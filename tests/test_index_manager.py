from unittest.mock import Mock

import pytest
from azure.core.exceptions import ResourceExistsError
from azure.search.documents.indexes import SearchIndexClient

from search.index_manager import ensure_document_index
from search.index_schema import (
    EMBEDDING_DIMENSIONS,
    INDEX_NAME,
    build_document_index,
)


def test_create_document_index():
    client = Mock(spec=SearchIndexClient)

    expected_index = build_document_index()

    client.create_index.return_value = expected_index

    result = ensure_document_index(client)

    assert result.name == INDEX_NAME

    client.create_index.assert_called_once()

    created_index = client.create_index.call_args.args[0]

    assert created_index.name == INDEX_NAME

    client.get_index.assert_not_called()


def test_existing_document_index():
    client = Mock(spec=SearchIndexClient)

    existing_index = build_document_index()

    client.create_index.side_effect = ResourceExistsError(
        message="Index already exists"
    )

    client.get_index.return_value = existing_index

    result = ensure_document_index(client)

    assert result.name == INDEX_NAME

    client.create_index.assert_called_once()
    client.get_index.assert_called_once_with(INDEX_NAME)


def test_existing_index_invalid_dimensions():
    client = Mock(spec=SearchIndexClient)

    existing_index = build_document_index()

    vector_field = next(
        field for field in existing_index.fields if field.name == "content_vector"
    )

    vector_field.vector_search_dimensions = EMBEDDING_DIMENSIONS - 1

    client.create_index.side_effect = ResourceExistsError(
        message="Index already exists"
    )

    client.get_index.return_value = existing_index

    with pytest.raises(
        RuntimeError,
        match="incompatible vector dimensions",
    ):
        ensure_document_index(client)


def test_existing_index_missing_field():
    client = Mock(spec=SearchIndexClient)

    existing_index = build_document_index()

    existing_index.fields = [
        field for field in existing_index.fields if field.name != "content"
    ]

    client.create_index.side_effect = ResourceExistsError(
        message="Index already exists"
    )

    client.get_index.return_value = existing_index

    with pytest.raises(
        RuntimeError,
        match="missing field",
    ):
        ensure_document_index(client)
