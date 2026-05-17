import re
import json
import os
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from ddgs import DDGS
from wiki import (
    wiki_search, wiki_get_page, wiki_get_page_by_path,
    wiki_create_page, wiki_update_page, wiki_list_pages
)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
INDEX_PATH = "schema/index"
SCHEMA_PATH = "schema/agents"
SCHEMA_PATHS = {INDEX_PATH, SCHEMA_PATH}

app = FastAPI(title="LLM MCP Hub")


class SearchRequest(BaseModel):
    query: str
    max_results: int = 5

class WikiQueryRequest(BaseModel):
    query: str
    max_results: int = 3

class WikiIngestRequest(BaseModel):
    title: str
    content: str

class WikiLintRequest(BaseModel):
    mode: str = "links"  # "links" | "index" | "full"


def _llm(prompt: str, model: str = None, timeout: float = 180.0) -> str:
    if model is None:
        model = os.environ.get("MAIN_MODEL", "gemma3:12b")
    r = httpx.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "stream": False, "options": {"temperature": 0.1}},
        timeout=timeout
    )
    return r.json()["message"]["content"]

def _parse_wiki_blocks(text: str) -> list:
    # Matches CREATE and UPDATE blocks separated by === headers ===
    # The [\s\S]+? is non-greedy multiline content; \Z anchors the last block
    blocks = []
    pattern = r'=== (CREATE|UPDATE): (.+?) ===\nTitle: (.+?)\n---\n([\s\S]+?)(?=\n=== |\Z)'
    for m in re.finditer(pattern, text):
        blocks.append({
            "action": m.group(1),
            "target": m.group(2).strip(),
            "title": m.group(3).strip(),
            "content": m.group(4).strip()
        })
    return blocks

def _get_index() -> str | None:
    try:
        page = wiki_get_page_by_path(INDEX_PATH)
        return page["content"] if page else None
    except Exception:
        return None

def _get_schema() -> dict | None:
    try:
        return wiki_get_page_by_path(SCHEMA_PATH)
    except Exception:
        return None

def _select_pages_from_index(query: str, index: str) -> list:
    # Uses the fast routing model to pick relevant page paths from the index
    # without loading full page content — keeps query latency low
    routing_model = os.environ.get("ROUTING_MODEL", "qwen3:8b")
    prompt = (
        f"Wiki index:\n{index}\n\n"
        f"Query: {query}\n\n"
        "Which 1-3 page paths are most relevant? "
        'Return only a JSON array of path strings, e.g. ["homelab/proxmox", "assistant/orchestrator"]'
    )
    try:
        content = _llm(prompt, model=routing_model, timeout=15.0)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        match = re.search(r'\[.*?\]', content, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return []

def _compact_index(index: str) -> str:
    lines = []
    title, path = "", ""
    for line in index.splitlines():
        if line.startswith("## "):
            title = line[3:].strip()
        elif line.startswith("Path: "):
            path = line[6:].strip()
            if title and path:
                lines.append(f"{path} | {title}")
                title, path = "", ""
    return "\n".join(lines)

def _build_index_content(all_pages: list) -> str:
    # Extracts the first non-heading line as a summary — cheap, no LLM call per page
    entries = []
    for page in all_pages:
        if page["path"] in SCHEMA_PATHS:
            continue
        try:
            content = wiki_get_page(page["id"])
            lines = [l for l in content.split("\n") if l.strip() and not l.startswith("#")]
            summary = lines[0][:150] if lines else ""
            entries.append(f"## {page['title']}\nPath: {page['path']}\n{summary}")
        except Exception:
            entries.append(f"## {page['title']}\nPath: {page['path']}")
    return "# Wiki Index\n\n" + "\n\n".join(entries)

def _lint_index() -> dict:
    all_pages = wiki_list_pages()
    index_content = _build_index_content(all_pages)
    existing = wiki_get_page_by_path(INDEX_PATH)
    if existing:
        ok, msg = wiki_update_page(existing["id"], "Wiki Index", INDEX_PATH, index_content)
    else:
        ok, msg = wiki_create_page("Wiki Index", INDEX_PATH, index_content)
    page_count = sum(1 for p in all_pages if p["path"] not in SCHEMA_PATHS)
    return {"mode": "index", "pages_indexed": page_count, "success": ok, "message": msg}

def _lint_links() -> dict:
    # Case-insensitive title matching — [[My Page]] and [[my page]] are treated as the same
    all_pages = wiki_list_pages()
    all_titles = {p["title"].lower() for p in all_pages}
    broken = []
    for page in all_pages:
        if page["path"] in SCHEMA_PATHS:
            continue
        try:
            content = wiki_get_page(page["id"])
            links = re.findall(r'\[\[([^\]]+)\]\]', content)
            for link in links:
                if link.lower() not in all_titles:
                    broken.append({
                        "page": page["title"],
                        "path": page["path"],
                        "broken_link": link
                    })
        except Exception:
            pass
    return {"mode": "links", "broken_links": broken, "total": len(broken)}

def _lint_full() -> dict:
    # Limited to first 20 pages to stay within the LLM context window
    link_result = _lint_links()
    all_pages = wiki_list_pages()
    previews = []
    for page in all_pages[:20]:
        if page["path"] in SCHEMA_PATHS:
            continue
        try:
            content = wiki_get_page(page["id"])
            previews.append(f"### {page['title']} ({page['path']})\n{content[:600]}")
        except Exception:
            pass
    prompt = (
        "You are a wiki quality checker. Review these wiki pages:\n\n"
        + "\n\n---\n\n".join(previews)
        + "\n\nFind: (1) contradictions between pages, "
        "(2) pages that should link to each other but currently don't, "
        "(3) outdated or suspicious content.\n"
        'Return a JSON object: {"contradictions": [...], "missing_links": [...], "warnings": [...]}'
    )
    try:
        response = _llm(prompt, timeout=180.0)
        match = re.search(r'\{[\s\S]+\}', response)
        analysis = json.loads(match.group()) if match else {"raw": response}
    except Exception as e:
        analysis = {"error": str(e)}
    return {**link_result, "mode": "full", "analysis": analysis}


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/tools/web_search")
def web_search(req: SearchRequest):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(req.query, max_results=req.max_results):
            results.append({"title": r["title"], "url": r["href"], "snippet": r["body"]})
    return {"results": results}

@app.post("/tools/wiki_query")
def wiki_query(req: WikiQueryRequest):
    # Index-first: precise LLM-guided selection; fallback to full-text search
    # when no index exists yet (e.g. first run before wiki_lint index)
    index = _get_index()
    if index:
        paths = _select_pages_from_index(req.query, index)
        results = []
        for path in paths[:req.max_results]:
            try:
                page = wiki_get_page_by_path(path)
                if page:
                    results.append({
                        "title": page["title"],
                        "path": page["path"],
                        "content": wiki_get_page(page["id"])
                    })
            except Exception:
                pass
        if results:
            return {"results": results}
    pages = wiki_search(req.query, req.max_results)
    context = []
    for page in pages:
        content = wiki_get_page(page["id"])
        context.append({"title": page["title"], "path": page["path"], "content": content})
    return {"results": context}

@app.post("/tools/wiki_ingest")
def wiki_ingest(req: WikiIngestRequest):
    index = _get_index()
    schema_page = _get_schema()
    related = wiki_search(req.title, limit=5)
    existing = []
    seen_ids = set()
    for page in related:
        if page["path"] in SCHEMA_PATHS:
            continue
        pid = page["id"]
        if pid not in seen_ids:
            seen_ids.add(pid)
            existing.append({
                "id": pid,
                "title": page["title"],
                "path": page["path"],
                "content": wiki_get_page(pid)
            })
    schema_section = f"WIKI SCHEMA (read-only, follow these conventions):\n{schema_page['content']}\n\n" if schema_page else ""
    index_section = f"WIKI INDEX (existing pages):\n{_compact_index(index)}\n\n" if index else ""
    existing_str = "\n\n".join(
        f"=== EXISTING id={p['id']} path={p['path']} ===\n{p['content']}"
        for p in existing
    ) if existing else "No related pages found."
    prompt = (
        "You are a wiki maintainer. Integrate new knowledge into the existing wiki.\n\n"
        + schema_section
        + index_section
        + "NEW CONTENT:\n"
        f"Title: {req.title}\n"
        f"{req.content}\n\n"
        "EXISTING RELATED PAGES (full content):\n"
        f"{existing_str}\n\n"
        "Instructions:\n"
        "- Use ONLY information explicitly stated in NEW CONTENT — do not add general knowledge\n"
        "- Always write notes in English regardless of the language of NEW CONTENT\n"
        "- Follow the Wiki Schema conventions exactly\n"
        "- One concept per note, no topic mix — split complex content into multiple atomic notes\n"
        "- Create new wiki pages for concepts not yet covered\n"
        "- Update existing pages to add [[wikilinks]] to related new pages\n"
        "- Use [[Page Title]] syntax for links between related pages\n"
        "- Avoid creating pages that already exist (check the index)\n"
        "- Preserve existing content, only add or integrate new information\n"
        "Output only blocks in this exact format, no other text:\n\n"
        "=== CREATE: path/slug ===\n"
        "Title: Page Title\n"
        "---\n"
        "Full markdown content with [[wikilinks]]\n\n"
        "=== UPDATE: <id> ===\n"
        "Title: Existing Page Title\n"
        "---\n"
        "Complete updated markdown content with new links added\n\n"
        "(<id> is the alphanumeric note ID from 'EXISTING id=...' lines, e.g. oGZbWpTlfI4Q)"
    )
    response = _llm(prompt, timeout=360.0)
    blocks = _parse_wiki_blocks(response)
    # Fallback: if LLM output is unparseable, create the page directly with given title
    if not blocks:
        path = req.title.lower().replace(" ", "-")
        ok, msg = wiki_create_page(req.title, path, req.content)
        _lint_index()
        return {"created": [req.title] if ok else [], "updated": [], "errors": [] if ok else [msg]}
    created, updated, errors = [], [], []
    for block in blocks:
        if block["action"] == "CREATE":
            ok, msg = wiki_create_page(block["title"], block["target"], block["content"])
            if ok:
                created.append(block["title"])
            else:
                errors.append(f"CREATE '{block['title']}': {msg}")
        elif block["action"] == "UPDATE":
            try:
                pid = block["target"]
                page_info = next((p for p in existing if p["id"] == pid), None)
                if not page_info:
                    errors.append(f"UPDATE '{block['title']}': unknown note id '{pid}' — skipped")
                    continue
                path = page_info["path"]
                ok, msg = wiki_update_page(pid, block["title"], path, block["content"])
                if ok:
                    updated.append(block["title"])
                else:
                    errors.append(f"UPDATE '{block['title']}': {msg}")
            except (ValueError, TypeError) as e:
                errors.append(f"UPDATE parse error: {e}")
    # Rebuild index after every ingest to keep it current
    _lint_index()
    return {"created": created, "updated": updated, "errors": errors}

@app.post("/tools/wiki_lint")
def wiki_lint(req: WikiLintRequest):
    if req.mode == "index":
        return _lint_index()
    elif req.mode == "links":
        return _lint_links()
    elif req.mode == "full":
        return _lint_full()
    return {"error": f"Unknown mode '{req.mode}'. Use 'index', 'links', or 'full'."}
