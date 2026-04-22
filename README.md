# Reading Assistant

An AI-powered reading companion for research papers and books.
Grounded Q&A, concept explanations, and cross-reference lookup —
built around structured PDF ingestion, retrieval-augmented generation,
and agent orchestration.

## Status

Work in progress. This repo is the active evolution of earlier work on
AI-powered study tools, refocused on the harder problem of deep reading
of scientific papers (tables, equations, citations, cross-references).

## Stack

- **PDF ingestion**: [Docling](https://github.com/docling-project/docling)
  for structure-aware parsing of scientific papers
- **Vector store**: Chroma
- **LLM**: Anthropic Claude (primary), OpenAI (fallback)
- **Orchestration**: LangGraph
- **Tools**: exposed via Model Context Protocol (MCP)
- **Backend**: FastAPI
- **Testing**: pytest

## Setup

```bash
uv sync
```

Drop a PDF into `samples/` and run the ingestion script (coming soon).

## Project structure

```
reading_assistant/
├── models/        # Pydantic schemas (Chunk, etc.)
├── ingest/        # Document ingestion pipeline (Docling + chunking)
├── retrieval/     # Vector store + retrieval logic
├── agent/         # LangGraph agent and tool integrations
└── api/           # FastAPI routes
tests/             # pytest suite
samples/           # (gitignored) test PDFs
```

## Author

Francesco — building this as part of my portfolio.
