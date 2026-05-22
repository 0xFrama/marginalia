from functools import lru_cache

from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

from models.chunk import Chunk
from models.embedding import EmbeddingRecord

DENSE_MODEL_NAME = "all-MiniLM-L6-v2"
SPARSE_MODEL_NAME = "Qdrant/bm25"


@lru_cache(maxsize=1)
def load_embed_model() -> SentenceTransformer:
    return SentenceTransformer(DENSE_MODEL_NAME)


@lru_cache(maxsize=1)
def load_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


def embed_text(text: str, model: SentenceTransformer | None = None) -> list[float]:
    embedder = model or load_embed_model()
    vector = embedder.encode(text, normalize_embeddings=True)
    return vector.tolist()


def sparse_embed_text(text: str) -> tuple[list[int], list[float]]:
    sparse_model = load_sparse_model()
    result = next(sparse_model.embed([text]))
    return result.indices.tolist(), result.values.tolist()


def embed_chunks(
    chunks: list[Chunk], model: SentenceTransformer | None = None
) -> list[EmbeddingRecord]:
    embedder = model or load_embed_model()
    texts = [chunk.text for chunk in chunks]
    dense_vectors = embedder.encode(texts, normalize_embeddings=True)

    sparse_model = load_sparse_model()
    sparse_results = list(sparse_model.embed(texts))

    return [
        EmbeddingRecord(
            chunk=chunk,
            embedding=vec.tolist(),
            sparse_indices=sp.indices.tolist(),
            sparse_values=sp.values.tolist(),
        )
        for chunk, vec, sp in zip(chunks, dense_vectors, sparse_results, strict=True)
    ]

