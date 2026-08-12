"""LangGraph + Ollama tracing.

Requires langgraph, langchain-ollama, and python-dotenv. Values load from sibling .env.
"""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

load_dotenv(Path(__file__).with_name(".env"))

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud"),
    base_url=os.getenv("OLLAMA_BASE_URL", "https://ollama.com"),
    client_kwargs={"headers": {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}},
)
graph = StateGraph(dict)
graph.add_node("answer", lambda state: {"answer": llm.invoke(state["question"]).content})
graph.add_edge(START, "answer")
graph.add_edge("answer", END)
app = graph.compile()

print(app.invoke({"question": "Explain observability in one sentence."})["answer"])
