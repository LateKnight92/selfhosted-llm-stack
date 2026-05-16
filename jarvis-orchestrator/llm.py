import httpx
import os
from typing import Generator

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("MAIN_MODEL", "gemma3:12b")
CODE_MODEL = os.environ.get("CODE_MODEL", "qwen2.5-coder:7b")
ASSISTANT_NAME = os.environ.get("ASSISTANT_NAME", "Assistant")

CODE_SYSTEM = f"You are {ASSISTANT_NAME}, an expert coding assistant. Write clean, well-structured code and explain your solution briefly."

def _build_messages(message: str, context: str, model: str) -> list:
    if model == CODE_MODEL:
        system = f"You are {ASSISTANT_NAME}, an expert coding assistant. Write clean, well-structured code and explain your solution briefly."
    elif context:
        system = f"You are {ASSISTANT_NAME}, a helpful personal assistant. Answer the user's question based on these search results:\n\n{context}"
    else:
        system = f"You are {ASSISTANT_NAME}, a helpful personal assistant."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": message}
    ]

def respond(message: str, context: str = "", model: str = MODEL) -> str:
    r = httpx.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": model, "messages": _build_messages(message, context, model), "stream": False},
        timeout=120.0
    )
    return r.json()["message"]["content"]

def stream_respond(message: str, context: str = "", model: str = MODEL) -> Generator[str, None, None]:
    with httpx.stream(
        "POST",
        f"{OLLAMA_URL}/api/chat",
        json={"model": model, "messages": _build_messages(message, context, model), "stream": True},
        timeout=120.0
    ) as r:
        for line in r.iter_lines():
            if line:
                yield line + "\n"
