from pydantic import BaseModel


class IndexingResult(BaseModel):
    source_file: str
    chunk_count: int
    indexed_count: int
    collection_name: str
