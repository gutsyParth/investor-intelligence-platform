from fastapi import FastAPI

from routes import health

app = FastAPI(
    title="Investor Intelligence Platform",
    description="AI-powered financial document intelligence using RAG",
    version="0.1.0",
)


app.include_router(health.router)
