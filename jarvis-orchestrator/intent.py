import httpx
import os
import re

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("ROUTING_MODEL", "qwen3:8b")

WIKI_KEYWORDS = [
    "was weiss ich", "was hab ich", "was habe ich",
    "habe ich notiert", "hab ich notiert",
    "steht in meinem wiki", "in meinem wiki",
    "meine notizen", "mein wiki",
    "habe ich dokumentiert", "hab ich dokumentiert",
]

CODE_KEYWORDS = [
    "schreib mir ein script", "schreib mir einen code",
    "schreib mir eine funktion", "schreib mir ein programm",
    "schreibe mir ein script", "schreibe mir einen code",
    "schreib mir eine klasse", "schreib mir einen",
    "debugge", "debug diesen", "fix diesen bug",
    "fix diesen fehler", "warum funktioniert dieser code nicht",
    "was ist falsch an diesem code",
    "erklaere diesen code", "erklaere mir diesen code",
    "was macht dieser code", "was macht diese funktion",
    "analysiere diesen code", "analysiere diese funktion",
    "ueberpruefe meinen code", "ueberpruefe diesen code",
    "review meinen code", "code review",
    "optimiere diesen code", "optimiere diese funktion",
    "refaktoriere", "refactor",
]

SYSTEM = """You are an intent classifier. Classify the user message into exactly one intent.

- chat: general questions, explanations, conversation
- web_search: needs current or real-time information (news, weather, prices, recent events)
- wiki_query: user asks what THEY personally know or have saved (personal notes, own documentation)
- code: user wants code written, debugged, reviewed, or explained line by line

Reply with only valid JSON: {"intent": "chat"} or {"intent": "web_search"} or {"intent": "wiki_query"} or {"intent": "code"}"""

def _normalize(text: str) -> str:
    return (text.lower()
        .replace("ß", "ss")
        .replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue"))

def classify(message: str) -> str:
    normalized = _normalize(message)

    for kw in WIKI_KEYWORDS:
        if kw in normalized:
            return "wiki_query"

    for kw in CODE_KEYWORDS:
        if kw in normalized:
            return "code"

    try:
        r = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": message}
                ],
                "stream": False,
                "options": {"temperature": 0}
            },
            timeout=30.0
        )
        content = r.json()["message"]["content"]
        match = re.search(r'\{"intent":\s*"(\w+)"\}', content)
        if match:
            return match.group(1)
    except Exception:
        pass
    return "chat"
