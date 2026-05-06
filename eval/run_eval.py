import json

from retrieval import Retriever
from index.qdrant_store import QdrantStore

with open("eval/questions.json", "r") as file:
    questions = json.load(file)

TOP_K = 5
matched = 0
total = len(questions)

if total == 0:
    raise RuntimeError("No evaluation questions found.")

store = QdrantStore()
retriever = Retriever(store=store)
for q in questions:
    hits = retriever.retrieve(q["question"], top_k=TOP_K)
    retrieved_ids = [hit.chunk.chunk_id for hit in hits]
    expected_ids = q["expected_chunk_ids"]
    success = any(expected_id in retrieved_ids for expected_id in expected_ids)
    if success:
        matched += 1
    status = "PASS" if success else "FAIL"
    print(f"{status} {q['question']}")
    print(f"  expected: {expected_ids}")
    print(f"  retrieved: {retrieved_ids}")

recall = matched / total
print(f"recall@{TOP_K}: {recall:.2f} ({matched}/{total})")
