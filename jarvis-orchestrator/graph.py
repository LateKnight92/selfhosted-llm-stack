from langgraph.graph import StateGraph, END
from typing import TypedDict
from intent import classify
from tools import web_search, wiki_query

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

def route(state: State) -> str:
    if state["intent"] == "web_search":
        return "web_search"
    if state["intent"] == "wiki_query":
        return "wiki_query"
    return "end"

def build_graph():
    g = StateGraph(State)
    g.add_node("classify", classify_node)
    g.add_node("web_search", web_search_node)
    g.add_node("wiki_query", wiki_query_node)
    g.set_entry_point("classify")
    g.add_conditional_edges("classify", route, {
        "web_search": "web_search",
        "wiki_query": "wiki_query",
        "end": END
    })
    g.add_edge("web_search", END)
    g.add_edge("wiki_query", END)
    return g.compile()
