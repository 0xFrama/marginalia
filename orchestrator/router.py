def route(question: str, llm) -> list[str]:

    system_prompt = (
        "You are a router for a clinical assistant. Your job is to decide which data source can answer a question.\n"
        "Two choices: patient or guideline.\n"
        "Patient: facts about one specific patient: their lab results, medications, conditions, visits. (e.g. 'what is this patient's latest HbA1c')\n"
        "Guideline: general medical knowledge from public clinical guideline, not tied to any patient. (e.g. 'what is the recommended HbA1c target for diabetics?')\n"
        "If the question needs both a patient's own data and general guidance, answer both. If you are unsure, answer both.\n"
        "reply with exactly one word and nothing else: patient, guideline, both."
    )

    user_prompt = f"Question: {question}"

    reply = llm.generate(system_prompt, user_prompt)

    cleaned = reply.strip().lower()

    if cleaned == "patient":
        return ["patient"]
    if cleaned == "guideline":
        return ["guideline"]
    return ["patient", "guideline"]
