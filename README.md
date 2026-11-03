# AI-Powered Investor Intelligence Platform

An AI-powered financial document analysis platform built
using FastAPI, PostgreSQL, Azure OpenAI, Azure AI Search,
and Retrieval-Augmented Generation (RAG).

## Project Status

Under active development.

## Current Features

- FastAPI application initialization
- Health check endpoint
- Automatic Swagger API documentation

## Tech Stack

- Python 3.11
- FastAPI
- Uvicorn

## Local Development

Create a virtual environment:

python3.11 -m venv .venv

Activate it:

source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Start the application:

uvicorn main:app --reload

## API Endpoints

- GET /health - Application health check
- GET /docs - Swagger API documentation

## Upcoming Features

- PostgreSQL integration
- Financial document ingestion
- PDF parsing and chunking
- Azure OpenAI embeddings
- Azure AI Search
- RAG-based question answering
- Docker containerization
- Kubernetes deployment
- GitHub Actions CI/CD

## Acknowledgements

Inspired by Sandesh Hase's AI-Powered Investor
Intelligence Platform tutorial.

This implementation is being developed independently,
step by step, for educational purposes.
