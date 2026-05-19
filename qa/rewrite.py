from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from models.chat import ChatMessage

REWRITE_SYSTEM_PROMPT = """You rewrite follow-up questions into standalone search queries for document retrieval.
Use the chat history only to resolve references and omitted context.
Return only the rewritten standalone query with no extra commentary.
If the current question is already standalone, return it unchanged.
"""

REWRITE_USER_PROMPT_TEMPLATE = """Current question:
{question}

Rewrite the current question as a standalone retrieval query."""


def _build_rewrite_history_messages(
    chat_history: list[ChatMessage] | None,
) -> list[BaseMessage]:
    if not chat_history:
        return []

    history_messages: list[BaseMessage] = []
    for message in chat_history:
        if message.role == "user":
            history_messages.append(HumanMessage(content=message.content))
        elif message.role == "assistant":
            history_messages.append(AIMessage(content=message.content))

    return history_messages


def build_rewrite_messages(
    question: str,
    chat_history: list[ChatMessage] | None = None,
) -> list[BaseMessage]:
    template = ChatPromptTemplate.from_messages(
        [
            ("system", REWRITE_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", REWRITE_USER_PROMPT_TEMPLATE),
        ]
    )

    return template.format_messages(
        chat_history=_build_rewrite_history_messages(chat_history),
        question=question,
    )


def normalize_rewritten_query(rewritten_query: str, fallback_question: str) -> str:
    normalized = rewritten_query.strip().strip('"').strip("'")
    return normalized or fallback_question


def rewrite_query(
    question: str,
    chat_history: list[ChatMessage] | None,
    llm_client,
) -> str:
    if not chat_history:
        return question

    messages = build_rewrite_messages(question, chat_history)

    if hasattr(llm_client, "generate_messages"):
        rewritten_query = llm_client.generate_messages(messages)
    else:
        user_prompt = REWRITE_USER_PROMPT_TEMPLATE.format(question=question)
        rewritten_query = llm_client.generate(REWRITE_SYSTEM_PROMPT, user_prompt)

    return normalize_rewritten_query(rewritten_query, question)
