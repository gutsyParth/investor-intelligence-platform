import os

import pytest
from sqlalchemy import text

from config.settings import get_settings
from database.connection import get_engine


@pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION_TESTS") != "1",
    reason="Live database integration tests are disabled by default",
)
def test_database_connection():
    settings = get_settings()

    if settings.db_password is None or not settings.db_password.get_secret_value():
        pytest.skip("Database credentials not configured")

    with get_engine().connect() as connection:
        result = connection.execute(text("SELECT current_database()"))
        database_name = result.scalar_one()

    assert database_name == settings.db_name
