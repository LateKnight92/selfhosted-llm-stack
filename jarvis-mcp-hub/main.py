from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from ddgs import DDGS
from wiki import wiki_search, wiki_get_page, wiki_create_page

app = FastAPI(title="Local LLM MCP Hub")

class SearchRequest(BaseModel):
    query: str
    max_results: int = 5

class WikiQueryRequest(BaseModel):
    query: str
    max_results: int = 3

class WikiIngestRequest(BaseModel):
    title: str
    path: str
    content: str
    locale: str = "en"

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
    pages = wiki_search(req.query, req.max_results)
    context = []
    for page in pages:
        content = wiki_get_page(int(page["id"]))
        context.append({"title": page["title"], "path": page["path"], "content": content})
    return {"results": context}

@app.post("/tools/wiki_ingest")
def wiki_ingest(req: WikiIngestRequest):
    success, message = wiki_create_page(req.title, req.path, req.content, req.locale)
    return {"success": success, "message": message}
