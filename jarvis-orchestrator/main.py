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

@app.post("/chat")
def chat(req: ChatRequest):
    intent, context = _route(req.message)
    return {"response": respond(req.message, context, _get_model(intent))}

@app.post("/api/chat")
def ollama_chat(req: OllamaRequest):
    message = req.messages[-1]["content"] if req.messages else ""
    intent, context = _route(message)
    model = _get_model(intent)

    if req.stream:
        return StreamingResponse(
            stream_respond(message, context, model),
            media_type="application/x-ndjson"
        )
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
