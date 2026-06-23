from typing import Tuple


def mask(text: str, identifiers: dict) -> Tuple[str, dict]:
    mapping = {}
    for label, value in identifiers.items():
        placeholder = f"[{label}]"
        if value and value in text:
            text = text.replace(value, placeholder)
            mapping[placeholder] = value
    return text, mapping


def unmask(text: str, mapping: dict) -> str:
    for placeholder, value in mapping.items():
        if placeholder and placeholder in text:
            text = text.replace(placeholder, value)
    return text
