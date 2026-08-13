"""LangGraph + Anthropic tracing.

Requires langgraph, langchain-anthropic, and python-dotenv. Values load from sibling .env.
"""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = ChatAnthropic(
    model="claude-3-5-haiku-latest",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=128,
)
graph = StateGraph(dict)
graph.add_node("answer", lambda state: {"answer": llm.invoke(state["question"]).content})
graph.add_edge(START, "answer")
graph.add_edge("answer", END)
app = graph.compile()

print(app.invoke({"question": "Explain observability in one sentence."})["answer"])
