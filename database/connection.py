from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine

from config.settings import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()

    if settings.db_password is None or not settings.db_password.get_secret_value():
        raise RuntimeError("DB_PASSWORD is not configured")

    database_url = URL.create(
        drivername="postgresql+psycopg",
        username=settings.db_user,
        password=settings.db_password.get_secret_value(),
        host=settings.db_host,
        port=settings.db_port,
        database=settings.db_name,
    )

    return create_engine(
        database_url,
        connect_args={"sslmode": "require"},
        pool_pre_ping=True,
    )
