import os

from openai import OpenAI, OpenAIError


class OpenAIService:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature

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
