import httpx
import os
import re
import json

MCP_HUB_URL = os.environ.get("MCP_HUB_URL", "http://localhost:8080")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

def _normalize(text: str) -> str:
    # Converts German umlauts to ASCII so keyword matching works
    # regardless of how the user types (ü vs ue, ß vs ss, etc.)
    return (text.lower()
        .replace("ß", "ss")
        .replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue"))

def _extract_search_terms(message: str) -> list:
    # Uses the routing model to extract 1-3 key terms from a natural language query
    # so wiki search finds relevant pages even for long questions
    try:
        r = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": os.environ.get("ROUTING_MODEL", "qwen3:8b"),
                "messages": [{"role": "user", "content": f"Extract the specific topic or entity name from this question for a personal wiki search. Return only a JSON array with 1-2 specific proper names or technical terms. Do NOT include generic words like 'specifications', 'performance', 'information', 'details'. Example: [\"GPU\"] or [\"Proxmox\", \"NAS\"]\n\nQuestion: {message}"}],
                "stream": False,
                "options": {"temperature": 0}
            },
            timeout=15.0
        )
        content = r.json()["message"]["content"]
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        match = re.search(r'\[.*?\]', content, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return [message]

def _extract_title(content: str) -> str:
    # Generates a short wiki page title from user-provided content
    # so wiki_ingest_from_message doesn't need a manually specified title
    try:
        r = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": os.environ.get("ROUTING_MODEL", "qwen3:8b"),
                "messages": [{"role": "user", "content": f"Generate a short wiki page title (3-6 words) for this content. Return only the title, nothing else.\n\nContent: {content}"}],
                "stream": False,
                "options": {"temperature": 0}
            },
            timeout=15.0
        )
        return r.json()["message"]["content"].strip().strip('"')
    except Exception:
        return content[:50]

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
        return f"[web_search failed: {e}]"

def wiki_lint(message: str) -> str:
    # Detects the lint mode from keywords in the user message:
    # "link" → check broken wikilinks, "full/komplett" → full analysis, default → rebuild index
    normalized = _normalize(message)
    if any(kw in normalized for kw in ["link", "verlinkt"]):
        mode = "links"
    elif any(kw in normalized for kw in ["vollstaendig", "komplett", "alles", "full"]):
        mode = "full"
    else:
        mode = "index"
    try:
        r = httpx.post(f"{MCP_HUB_URL}/tools/wiki_lint", json={"mode": mode}, timeout=300.0)
        result = r.json()
        if mode == "index":
            return f"Index built: {result.get('pages_indexed', '?')} pages indexed."
        elif mode == "links":
            total = result.get("total", 0)
            if total == 0:
                return "All links valid — no broken wikilinks found."
            broken = result.get("broken_links", [])
            lines = [f"- [{b['page']}]: [[{b['broken_link']}]]" for b in broken[:10]]
            suffix = f"\n_(first 10 of {total})_" if total > 10 else ""
            return f"{total} broken links:\n" + "\n".join(lines) + suffix
        elif mode == "full":
            total = result.get("total", 0)
            analysis = result.get("analysis", {})
            parts = [f"Link check: {total} broken links"]
            if analysis.get("contradictions"):
                parts.append("Contradictions:\n" + "\n".join(f"- {c}" for c in analysis["contradictions"][:5]))
            if analysis.get("missing_links"):
                parts.append("Missing links:\n" + "\n".join(f"- {l}" for l in analysis["missing_links"][:5]))
            if analysis.get("warnings"):
                parts.append("Warnings:\n" + "\n".join(f"- {w}" for w in analysis["warnings"][:5]))
            return "\n\n".join(parts)
    except Exception as e:
        return f"[wiki_lint failed: {e}]"

def wiki_ingest_from_message(message: str) -> str:
    # Strips the trigger phrase ("merke dir:", "notiere:", etc.) from the user message
    # so only the actual content is sent to the wiki
    content = re.sub(
        r'(?i)^(merke\s+dir|notiere(\s+das)?|lern\s+das|f[uu]ge\s+\w*\s*zum\s+wiki\s+hinzu|speichere\s+in\s+dein\s+wiki)[:\s]+',
        '', message
    ).strip() or message
    title = _extract_title(content)
    try:
        r = httpx.post(
            f"{MCP_HUB_URL}/tools/wiki_ingest",
            json={"title": title, "content": content},
            timeout=300.0
        )
        result = r.json()
        created = result.get("created", [])
        updated = result.get("updated", [])
        errors = result.get("errors", [])
        parts = []
        if created:
            parts.append(f"Created: {', '.join(created)}")
        if updated:
            parts.append(f"Updated: {', '.join(updated)}")
        if errors:
            parts.append(f"Errors: {', '.join(errors)}")
        return "\n".join(parts) if parts else "Saved."
    except Exception as e:
        return f"[wiki_ingest failed: {e}]"

def wiki_query(query: str) -> str:
    # Runs multiple searches (one per extracted term) and deduplicates by path
    # so compound topics return results from both searches
    try:
        terms = _extract_search_terms(query)
        seen_paths = set()
        all_results = []
        for term in terms:
            r = httpx.post(
                f"{MCP_HUB_URL}/tools/wiki_query",
                json={"query": term},
                timeout=60.0
            )
            for res in r.json().get("results", []):
                path = res.get("path")
                if path not in seen_paths:
                    seen_paths.add(path)
                    all_results.append(res)
        if not all_results:
            return "[No wiki pages found]"
        return "\n\n---\n\n".join(
            f"## {res.get('title', '')}\n\n{res.get('content', '')}"
            for res in all_results[:3]
        )
    except Exception as e:
        return f"[wiki_query failed: {e}]"
