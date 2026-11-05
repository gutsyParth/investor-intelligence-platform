from search.index_schema import (
    EMBEDDING_DIMENSIONS,
    INDEX_NAME,
    build_document_index,
)


def test_document_index_schema():
    index = build_document_index()

    assert index.name == INDEX_NAME

    fields = {field.name: field for field in index.fields}

    assert len(fields) == 6
    assert fields["id"].key is True

    assert fields["content_vector"].vector_search_dimensions == EMBEDDING_DIMENSIONS

    assert fields["content_vector"].vector_search_profile_name == "chunk-vector-profile"

    assert index.vector_search is not None
    assert len(index.vector_search.algorithms) == 1
    assert len(index.vector_search.profiles) == 1
