"""LangGraph + DeepSeek tracing. Requires langgraph, langchain-openai, and python-dotenv."""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = ChatOpenAI(model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"), api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1")
graph = StateGraph(dict)
graph.add_node("answer", lambda state: {"answer": llm.invoke(state["question"]).content})
graph.add_edge(START, "answer")
graph.add_edge("answer", END)
print(graph.compile().invoke({"question": "Explain observability in one sentence."})["answer"])
