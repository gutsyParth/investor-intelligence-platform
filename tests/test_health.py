from collections.abc import Generator
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from database.session import get_db
from main import app

client = TestClient(app)


@pytest.fixture
def mock_db() -> Generator[Mock, None, None]:
    session = Mock(spec=Session)

    app.dependency_overrides[get_db] = lambda: session

    try:
        yield session
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_database_health_success(mock_db: Mock):
    response = client.get("/health/db")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

    mock_db.execute.assert_called_once()


def test_database_health_failure(mock_db: Mock):
    mock_db.execute.side_effect = SQLAlchemyError(
        "Simulated database connection failure"
    )

    response = client.get("/health/db")

    assert response.status_code == 503
    assert response.json() == {"detail": "Database unavailable"}
