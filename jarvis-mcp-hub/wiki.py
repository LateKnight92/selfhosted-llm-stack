import httpx
import os

TRILIUM_URL = os.environ.get("TRILIUM_URL", "http://localhost:8080/etapi")
WIKI_LABEL = "jarvisWiki"

def _headers() -> dict:
    # v0.93.0+: Bearer token required
    return {"Authorization": f"Bearer {os.environ.get('TRILIUM_API_TOKEN', '')}"}

def _json_headers() -> dict:
    return {**_headers(), "Content-Type": "application/json"}

def _search_notes(query: str, limit: int = 10) -> list:
    r = httpx.get(
        f"{TRILIUM_URL}/notes",
        params={"search": query, "limit": limit},
        headers=_headers(),
        timeout=10.0
    )
    return r.json().get("results", [])

def _get_attrs(note_id: str) -> dict:
    # Attributes are embedded in GET /etapi/notes/{noteId} — no separate endpoint exists
    try:
        r = httpx.get(f"{TRILIUM_URL}/notes/{note_id}", headers=_headers(), timeout=5.0)
        return {a["name"]: a["value"] for a in r.json().get("attributes", []) if a.get("type") == "label"}
    except Exception:
        return {}

def _add_attr(note_id: str, name: str, value: str) -> None:
    # Trilium ETAPI: attributes are created via POST /etapi/attributes with noteId in body
    httpx.post(
        f"{TRILIUM_URL}/attributes",
        json={"noteId": note_id, "type": "label", "name": name, "value": value},
        headers=_json_headers(),
        timeout=5.0
    )

def _is_wiki_note(note_id: str) -> tuple[bool, dict]:
    # ETAPI text search does not filter by label reliably — check client-side
    attrs = _get_attrs(note_id)
    return WIKI_LABEL in attrs, attrs

def wiki_search(query: str, limit: int = 5) -> list:
    # Fetch extra to account for client-side filtering of non-wiki notes
    results = _search_notes(query, limit * 4)
    out = []
    for n in results:
        is_wiki, attrs = _is_wiki_note(n["noteId"])
        if not is_wiki:
            continue
        out.append({
            "id": n["noteId"],
            "title": n["title"],
            "path": attrs.get("wikiPath", n["noteId"]),
            "description": ""
        })
        if len(out) >= limit:
            break
    return out

def wiki_get_page(note_id: str) -> str:
    content_r = httpx.get(f"{TRILIUM_URL}/notes/{note_id}/content", headers=_headers(), timeout=10.0)
    meta_r = httpx.get(f"{TRILIUM_URL}/notes/{note_id}", headers=_headers(), timeout=10.0)
    title = meta_r.json().get("title", "")
    return f"# {title}\n\n{content_r.text}"

def wiki_create_page(title: str, path: str, content: str, locale: str = "en") -> tuple:
    try:
        r = httpx.post(
            f"{TRILIUM_URL}/create-note",
            json={"parentNoteId": "root", "title": title, "type": "code", "mime": "text/markdown", "content": content},
            headers=_json_headers(),
            timeout=10.0
        )
        note_id = r.json()["note"]["noteId"]
        _add_attr(note_id, WIKI_LABEL, "true")
        _add_attr(note_id, "wikiPath", path)
        return True, note_id
    except Exception as e:
        return False, str(e)

def wiki_update_page(note_id: str, title: str, path: str, content: str) -> tuple:
    try:
        r = httpx.put(
            f"{TRILIUM_URL}/notes/{note_id}/content",
            content=content.encode(),
            headers={**_headers(), "Content-Type": "text/plain"},
            timeout=10.0
        )
        r.raise_for_status()
        httpx.patch(f"{TRILIUM_URL}/notes/{note_id}", json={"title": title}, headers=_json_headers(), timeout=10.0)
        return True, ""
    except Exception as e:
        return False, str(e)

def wiki_list_pages(limit: int = 200) -> list:
    # Fetch broadly and filter client-side — ETAPI label filter in search is unreliable
    results = _search_notes(f"#{WIKI_LABEL}", limit * 2)
    out = []
    for n in results:
        is_wiki, attrs = _is_wiki_note(n["noteId"])
        if not is_wiki:
            continue
        out.append({"id": n["noteId"], "title": n["title"], "path": attrs.get("wikiPath", n["noteId"])})
        if len(out) >= limit:
            break
    return out

def wiki_get_page_by_path(path: str, locale: str = "en") -> dict | None:
    # Use wiki_list_pages (client-side label check) instead of search label filter
    # because ETAPI label filter in search queries is unreliable
    for page in wiki_list_pages():
        if page["path"] == path:
            content_r = httpx.get(f"{TRILIUM_URL}/notes/{page['id']}/content", headers=_headers(), timeout=10.0)
            return {"id": page["id"], "title": page["title"], "path": path, "content": content_r.text}
    return None
