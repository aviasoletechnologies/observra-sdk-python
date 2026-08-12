"""LangGraph + OpenRouter tracing.

Requires langgraph, langchain-openai, and python-dotenv. Values load from sibling .env.
"""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv(Path(__file__).with_name(".env"))

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "cohere/north-mini-code:free"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://api.openrouter.ai/api/v1",
)
graph = StateGraph(dict)
graph.add_node("answer", lambda state: {"answer": llm.invoke(state["question"]).content})
graph.add_edge(START, "answer")
graph.add_edge("answer", END)
app = graph.compile()

print(app.invoke({"question": "Explain observability in one sentence."})["answer"])
