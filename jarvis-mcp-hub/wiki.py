import httpx
import os

WIKI_URL = os.environ.get("WIKI_URL", "http://localhost:3000/graphql")

def _headers() -> dict:
    token = os.environ.get("WIKI_API_TOKEN", "")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

def wiki_search(query: str, limit: int = 5) -> list:
    gql = {
        "query": "query Search($query: String!) { pages { search(query: $query) { results { id title description path } } } }",
        "variables": {"query": query}
    }
    r = httpx.post(WIKI_URL, json=gql, headers=_headers(), timeout=10.0)
    results = r.json()["data"]["pages"]["search"]["results"]
    return results[:limit]

def wiki_get_page(page_id: int) -> str:
    gql = {
        "query": "query GetPage($id: Int!) { pages { single(id: $id) { id title content } } }",
        "variables": {"id": page_id}
    }
    r = httpx.post(WIKI_URL, json=gql, headers=_headers(), timeout=10.0)
    page = r.json()["data"]["pages"]["single"]
    return f"# {page['title']}\n\n{page['content']}"

def wiki_create_page(title: str, path: str, content: str, locale: str = "en") -> bool:
    gql = {
        "query": 'mutation CreatePage($content: String!, $path: String!, $title: String!, $locale: String!) { pages { create(content: $content description: "" editor: "markdown" isPrivate: false isPublished: true locale: $locale path: $path tags: [] title: $title) { responseResult { succeeded message } } } }',
        "variables": {"content": content, "path": path, "title": title, "locale": locale}
    }
    r = httpx.post(WIKI_URL, json=gql, headers=_headers(), timeout=10.0)
    result = r.json()["data"]["pages"]["create"]["responseResult"]
    return result["succeeded"], result.get("message", "")
