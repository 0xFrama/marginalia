import json
from pathlib import Path

from models.chunk import Chunk


def export_to_jsonl(chunks: list[Chunk], output_path: str | Path) -> None:
    path = Path(output_path)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            json.dump(chunk.to_record(), f, ensure_ascii=False)
            f.write("\n")
