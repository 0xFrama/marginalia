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

Tune retrieval at the command line:

```bash
uv run python -m expl.ask "What is attention?" --top-k 5 --min-score 0.55
```

Enable cross-encoder reranking:

```bash
uv run python -m expl.ask "What is attention?" --candidate-k 10 --top-k 3 --rerank
```

Ask a follow-up question with request-level chat history:

```json
[
  {
    "role": "user",
    "content": "What is attention?"
  },
  {
    "role": "assistant",
    "content": "Attention maps a query and key-value pairs to an output [1]."
  }
]
```

Save that JSON to a file such as `history.json`, then run:

```bash
uv run python -m expl.ask "Can you explain it more simply?" --history-file history.json
```

## API Usage

Qdrant must be running before starting the API. The `.env` file must contain `OPENAI_API_KEY`.

Start the FastAPI server:

```bash
uv run uvicorn api.app:app --reload
```

Index a local PDF through the API.

Nushell:

```nu
{ pdf_path: "samples/attention.pdf" } | to json | http post http://127.0.0.1:8000/index --content-type application/json
```

Ask a question through the API.

Nushell:

```nu
{ question: "What is attention?", top_k: 3, min_score: 0.55 } | to json | http post http://127.0.0.1:8000/ask --content-type application/json
```

Ask with reranking enabled.

Nushell:

```nu
{ question: "What is attention?", candidate_k: 10, top_k: 3, use_reranker: true } | to json | http post http://127.0.0.1:8000/ask --content-type application/json
```

Ask a follow-up question with prior chat turns.

Nushell:

```nu
{
  question: "Can you explain it more simply?"
  chat_history: [
    {
      role: "user"
      content: "What is attention?"
    }
    {
      role: "assistant"
      content: "Attention maps a query and key-value pairs to an output [1]."
    }
  ]
} | to json | http post http://127.0.0.1:8000/ask --content-type application/json
```

## Evaluation

Qdrant must already contain the indexed PDF before running evaluation.

Run retrieval evaluation:

```bash
uv run python -m eval.run_eval
```

The evaluator checks whether at least one expected chunk appears in the top-k retrieved results.

Current local result on `attention.pdf`:

```text
recall@5: 1.00 (6/6)
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
- Cited-source extraction from generated answers
- Grounded QA prompt builder
- OpenAI-based answer generation
- Request-level conversation memory through `chat_history`
- Manual end-to-end QA script
- FastAPI endpoints for path-based indexing and question answering
- Unit and integration tests for core layers

## Planned Work

- PDF file upload endpoint
- Persistent conversation memory beyond request-level chat history
- Evaluation set for retrieval and answer quality
- Documentation of architecture and design tradeoffs
