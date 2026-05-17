import json
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime, timezone
from graph import build_graph
from llm import respond, stream_respond, MODEL, CODE_MODEL

app = FastAPI()
graph = build_graph()

# These intents return the tool result directly — no LLM synthesis needed
# because the MCP hub already did the work (lint report, ingest status)
DIRECT_INTENTS = {"wiki_lint", "wiki_ingest"}

class ChatRequest(BaseModel):
    message: str

class OllamaRequest(BaseModel):
    model: str
    messages: list
    stream: bool = False

def _route(message: str) -> tuple[str, str]:
    result = graph.invoke({"message": message, "intent": "", "context": ""})
    return result["intent"], result.get("context", "")

def _get_model(intent: str) -> str:
    return CODE_MODEL if intent == "code" else MODEL

def _stream_direct(text: str, model_name: str):
    # Wraps a plain text result in the Ollama streaming format
    # so Open WebUI renders it correctly without waiting for a real LLM stream
    yield json.dumps({
        "model": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "message": {"role": "assistant", "content": text},
        "done": False
    }) + "\n"
    yield json.dumps({
        "model": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop"
    }) + "\n"

@app.post("/chat")
def chat(req: ChatRequest):
    intent, context = _route(req.message)
    if intent in DIRECT_INTENTS:
        return {"response": context or "Done."}
    return {"response": respond(req.message, context, _get_model(intent))}

@app.post("/api/chat")
def ollama_chat(req: OllamaRequest):
    message = req.messages[-1]["content"] if req.messages else ""
    intent, context = _route(message)

    if intent in DIRECT_INTENTS:
        response_text = context or "Done."
        if req.stream:
            return StreamingResponse(_stream_direct(response_text, req.model), media_type="application/x-ndjson")
        return {
            "model": req.model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message": {"role": "assistant", "content": response_text},
            "done": True,
            "done_reason": "stop"
        }

    model = _get_model(intent)
    if req.stream:
        return StreamingResponse(stream_respond(message, context, model), media_type="application/x-ndjson")
    return {
        "model": req.model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "message": {"role": "assistant", "content": respond(message, context, model)},
        "done": True,
        "done_reason": "stop"
    }

@app.get("/api/tags")
def tags():
    return {"models": [{"name": "assistant", "model": "assistant"}]}
