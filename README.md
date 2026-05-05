# Reading Assistant

A work-in-progress RAG reading assistant for technical and scientific documents.

Current implementation:

```text
PDF -> Docling -> chunks -> embeddings -> Qdrant -> retrieval -> evidence -> OpenAI answer
```

## Current Stack

- **PDF ingestion**: Docling
- **Embeddings**: sentence-transformers
- **Vector store**: Qdrant
- **LLM**: OpenAI
- **Data models**: Pydantic
- **Testing**: pytest
- **Environment management**: uv

## Setup

Install dependencies:

```bash
uv sync
```

Create a `.env` file with:

```text
OPENAI_API_KEY=your_api_key_here
```

Start Qdrant locally:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

For persistent local Qdrant storage:

```bash
docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

## Usage

Place a PDF at:

```text
samples/attention.pdf
```

Index the PDF into Qdrant:

```bash
uv run python -m index.pipeline
```

Ask a question:

```bash
uv run python -m expl.ask "What is attention?"
```

## Project Structure

```text
models/        Pydantic schemas for chunks, embeddings, retrieval hits, evidence, answers
ingest/        Docling-based PDF ingestion and JSONL export
index/         Embedding generation, Qdrant store, indexing pipeline
retrieval/     Retriever policy and evidence formatting
qa/            Prompt building, answer coordination, OpenAI wrapper
expl/          Manual exploration and smoke-test scripts
tests/         pytest suite
samples/       Local PDFs, gitignored
```

## Implemented Features

- Structure-aware PDF ingestion with Docling
- Typed chunk model with page, section, source, and chunk-type metadata
- JSONL export for chunks
- Local embeddings with sentence-transformers
- Qdrant indexing and similarity search
- Chunk-type filtering during retrieval
- Evidence block formatting with citation IDs
- Grounded QA prompt builder
- OpenAI-based answer generation
- Manual end-to-end QA script
- Unit and integration tests for core layers

## Planned Work

- FastAPI endpoints for indexing and asking questions
- Better retrieval controls, such as score thresholds and reranking
- Source filtering so final output distinguishes cited sources from retrieved evidence
- Conversation memory
- Evaluation set for retrieval and answer quality
- Documentation of architecture and design tradeoffs
