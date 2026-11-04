from collections.abc import Generator

from sqlalchemy.orm import Session

from database.connection import get_engine


def get_db() -> Generator[Session, None, None]:
    with Session(get_engine()) as session:
        yield session
