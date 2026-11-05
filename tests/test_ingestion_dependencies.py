from unittest.mock import patch

from routes.ingestion import get_document_ingestor


def test_ingestion_dependency_wiring():
    with (
        patch("routes.ingestion.get_embedding_client") as mock_embedding_client,
        patch("routes.ingestion.get_search_client") as mock_search_client,
        patch("routes.ingestion.generate_embeddings") as mock_generate,
        patch("routes.ingestion.ingest_pdf") as mock_pipeline,
    ):
        mock_generate.return_value = [[0.1] * 1536]

        # Create ingestion function
        ingestor = get_document_ingestor()

        # Execute ingestion function
        ingestor(
            b"%PDF-test",
            "report.pdf",
        )

        # Verify pipeline invocation
        mock_pipeline.assert_called_once()

        arguments = mock_pipeline.call_args.kwargs

        assert arguments["pdf_bytes"] == b"%PDF-test"
        assert arguments["filename"] == "report.pdf"

        # Verify search client injection
        assert arguments["search_client"] is mock_search_client.return_value

        # Verify embedding function injection
        embedding_function = arguments["embed_chunks"]

        result = embedding_function(["Sample financial text"])

        assert result == [[0.1] * 1536]

        mock_generate.assert_called_once_with(
            ["Sample financial text"],
            client=mock_embedding_client.return_value,
        )
