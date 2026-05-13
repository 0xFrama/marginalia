from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from models.chat import ChatMessage

SYSTEM_PROMPT = """You are a grounded assistant for technical agricultural and scientific documents.
    Answer only using the provided evidence.
    Cite evidence using citation IDs like [1] or [2].
    If the evidence is insufficient, say that the provided documents do not contain enough information.
    Do not invent facts or cite sources that are not provided.
"""

USER_PROMPT_TEMPLATE = """Evidence:
{evidence_context}

Question:
{question}

Answer:
Write a concise answer using only the evidence above. Include citations like [1] where relevant.
"""


def _build_chat_history_messages(
    chat_history: list[ChatMessage] | None,
) -> list[BaseMessage]:
    if not chat_history:
        return []

    chat_history_messages: list[BaseMessage] = []
    for message in chat_history:
        if message.role == "user":
            chat_history_messages.append(HumanMessage(content=message.content))
        elif message.role == "assistant":
            chat_history_messages.append(AIMessage(content=message.content))

    return chat_history_messages


def build_qa_messages(
    question: str, evidence_context: str, chat_history: list[ChatMessage] | None = None
) -> list[BaseMessage]:
    if not evidence_context.strip():
        evidence_context = "No evidence was retrieved."

    template = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", USER_PROMPT_TEMPLATE),
        ]
    )

    return template.format_messages(
        chat_history=_build_chat_history_messages(chat_history),
        evidence_context=evidence_context,
        question=question,
    )


def build_qa_prompt(
    question: str,
    evidence_context: str,
    chat_history: list[ChatMessage] | None = None,
) -> str:
    if not evidence_context.strip():
        evidence_context = "No evidence was retrieved."

    history_lines = []
    for message in chat_history or []:
        role = "User" if message.role == "user" else "Assistant"
        history_lines.append(f"{role}: {message.content}")

    chat_history_block = (
        "\n".join(history_lines) if history_lines else "No previous conversation."
    )

    return f"""Chat history:
{chat_history_block}

Evidence:
{evidence_context}

Question:
{question}

Answer:
Write a concise answer using only the evidence above. Include citations like [1] where relevant.
"""
