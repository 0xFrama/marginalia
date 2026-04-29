from functools import lru_cache

from sentence_transformers import SentenceTransformer

from models.chunk import Chunk
from models.embedding import EmbeddingRecord

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_embed_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed_text(text: str, model: SentenceTransformer | None = None) -> list[float]:
    embedder = model or load_embed_model()
    vector = embedder.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_chunks(
    chunks: list[Chunk], model: SentenceTransformer | None = None
) -> list[EmbeddingRecord]:
    embedder = model or load_embed_model()
    texts = [chunk.text for chunk in chunks]
    vectors = embedder.encode(texts, normalize_embeddings=True)

    return [
        EmbeddingRecord(chunk=chunk, embedding=vector.tolist())
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]

