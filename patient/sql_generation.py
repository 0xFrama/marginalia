def generate_sql(question, dictionary, patient_id, llm):
    system_prompt = (
        "You are an expert Text-to-SQL translator for a healthcare database.\n"
        "Your sole task is to translate natural language clinician questions into executable, syntactically correct SQLite queries.\n\n"
        "--- CRITICAL GUARDRAILS & SAFETY RULES ---\n"
        "1. ONLY return a single valid SELECT or WITH query. Do NOT write multi-statement scripts (no semicolons separating multiple queries).\n"
        "2. Never alter data. Commands like INSERT, UPDATE, DELETE, DROP, or CREATE are strictly forbidden.\n"
        "3. Output raw SQL code blocks only. Do NOT include markdown explanations, conversation, or greeting text outside the code block.\n"
        "4. If a question cannot be answered using the provided schema dictionary, return exactly: '-- ERROR: Insufficient schema context.'\n\n"
        "--- SQL COERCION & FILTERING BEST PRACTICES ---\n"
        "1. Chronological Filters: When asked for 'latest', 'most recent', or 'current' metrics, you MUST sort the query by time (e.g., ORDER BY taken_at DESC) and limit the result (LIMIT 1) per patient.\n"
        "2. Joins: Explicitly join tables on their specific foreign key references. Do not hallucinate columns.\n"
        "3. Case Sensitivity: Treat text matches using appropriate filtering functions if the data dictionary indicates categorical strings.\n\n"
        "--- AVAILABLE DATABASE SCHEMA DICTIONARY ---\n"
        f"Use the following definitions to link user text terms to exact database components: {dictionary}\n"
    )

    user_prompt = (
        f"Context Parameter:\n"
        f" - target_patient_id = {patient_id}\n\n"
        f"Question:\n"
        f"\n{question}\n"
        f"Generate the exact SQLite query to answer the question above utilizing the target_patient_id parameter."
    )

    return llm.generate(system_prompt, user_prompt)
