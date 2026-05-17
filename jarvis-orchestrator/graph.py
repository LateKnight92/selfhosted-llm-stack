from langgraph.graph import StateGraph, END
from typing import TypedDict
from intent import classify
from tools import web_search, wiki_query, wiki_lint, wiki_ingest_from_message

class State(TypedDict):
    message: str
    intent: str
    context: str

def classify_node(state: State) -> dict:
    return {"intent": classify(state["message"])}

def web_search_node(state: State) -> dict:
    return {"context": web_search(state["message"])}

def wiki_query_node(state: State) -> dict:
    return {"context": wiki_query(state["message"])}

def wiki_lint_node(state: State) -> dict:
    return {"context": wiki_lint(state["message"])}

def wiki_ingest_node(state: State) -> dict:
    return {"context": wiki_ingest_from_message(state["message"])}

def route(state: State) -> str:
    intent = state["intent"]
    if intent == "web_search":
        return "web_search"
    if intent == "wiki_query":
        return "wiki_query"
    if intent == "wiki_lint":
        return "wiki_lint"
    if intent == "wiki_ingest":
        return "wiki_ingest"
    return "end"

def build_graph():
    g = StateGraph(State)
    g.add_node("classify", classify_node)
    g.add_node("web_search", web_search_node)
    g.add_node("wiki_query", wiki_query_node)
    g.add_node("wiki_lint", wiki_lint_node)
    g.add_node("wiki_ingest", wiki_ingest_node)
    g.set_entry_point("classify")
    g.add_conditional_edges("classify", route, {
        "web_search": "web_search",
        "wiki_query": "wiki_query",
        "wiki_lint": "wiki_lint",
        "wiki_ingest": "wiki_ingest",
        "end": END
    })
    g.add_edge("web_search", END)
    g.add_edge("wiki_query", END)
    g.add_edge("wiki_lint", END)
    g.add_edge("wiki_ingest", END)
    return g.compile()
