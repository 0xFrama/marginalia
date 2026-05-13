import os

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from openai import OpenAI, OpenAIError


class OpenAIService:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature

    def generate_messages(self, messages: list[BaseMessage]) -> str:
        openai_messages = []

        for message in messages:
            if isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                raise RuntimeError(
                    f"Unsupported LangChain message type: {type(message).__name__}"
                )

            openai_messages.append({"role": role, "content": message.content})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                temperature=self.temperature,
            )
            return response.choices[0].message.content or ""
        except OpenAIError as e:
            raise RuntimeError(f"OpenAI generation failed: {e}") from e

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
            )
            return response.choices[0].message.content or ""
        except OpenAIError as e:
            raise RuntimeError(f"OpenAI generation failed: {e}") from e
