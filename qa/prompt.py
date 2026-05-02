SYSTEM_PROMPT = """You are a grounded assistant for technical agricultural and scientific documents.
    Answer only using the provided evidence.
    Cite evidence using citation IDs like [1] or [2].
    If the evidence is insufficient, say that the provided documents do not contain enough information.
    Do not invent facts or cite sources that are not provided.
    """


def build_qa_prompt(question: str, evidence_context: str) -> str:
    if not evidence_context.strip():
        evidence_context = "No evidence was retrieved."
    return f""" Evidence:
                {evidence_context}

                Question:
                {question} 

                Answer:
                Write a concise answer using only the evidence above. Include citations like [1] where relevant.
                """
