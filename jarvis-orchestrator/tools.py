import httpx
import os
import re
import json

MCP_HUB_URL = os.environ.get("MCP_HUB_URL", "http://localhost:8080")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

def _extract_search_terms(message: str) -> list:
    try:
        r = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": os.environ.get("ROUTING_MODEL", "qwen3:8b"),
                "messages": [{"role": "user", "content": f"Extract 1-3 key search terms from this question for searching a personal wiki. Return only a JSON array of strings, nothing else. Example: [\"NAS\", \"PVE Cluster\"]\n\nQuestion: {message}"}],
                "stream": False,
                "options": {"temperature": 0}
            },
            timeout=15.0
        )
        content = r.json()["message"]["content"]
        match = re.search(r'\[.*?\]', content, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return [message]

def web_search(query: str) -> str:
    try:
        r = httpx.post(
            f"{MCP_HUB_URL}/tools/web_search",
            json={"query": query},
            timeout=15.0
        )
        results = r.json().get("results", [])
        return "\n\n".join(
            f"{res.get('title', '')}\n{res.get('snippet', '')}"
            for res in results[:3]
        )
    except Exception as e:
        return f"[web_search fehlgeschlagen: {e}]"

def wiki_query(query: str) -> str:
    try:
        terms = _extract_search_terms(query)
        seen_paths = set()
        all_results = []

        for term in terms:
            r = httpx.post(
                f"{MCP_HUB_URL}/tools/wiki_query",
                json={"query": term},
                timeout=15.0
            )
            for res in r.json().get("results", []):
                path = res.get("path")
                if path not in seen_paths:
                    seen_paths.add(path)
                    all_results.append(res)

        if not all_results:
            return "[Keine Wiki-Seiten gefunden]"

        return "\n\n---\n\n".join(
            f"## {res.get('title', '')}\n\n{res.get('content', '')}"
            for res in all_results[:3]
        )
    except Exception as e:
        return f"[wiki_query fehlgeschlagen: {e}]"
