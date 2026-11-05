from fastapi import FastAPI

from config.settings import get_settings
from routes import documents, health, ingestion, questions

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="AI-powered financial document intelligence using RAG",
    version=settings.app_version,
    debug=settings.debug,
)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(ingestion.router)
app.include_router(questions.router)
