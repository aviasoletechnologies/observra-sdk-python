"""LangGraph + Hugging Face tracing. Requires langgraph, langchain-openai, and python-dotenv."""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = ChatOpenAI(model=os.getenv("HUGGINGFACE_MODEL", "meta-llama/Llama-3.1-8B-Instruct"), api_key=os.getenv("HUGGINGFACE_API_KEY"), base_url="https://router.huggingface.co/v1")
graph = StateGraph(dict)
graph.add_node("answer", lambda state: {"answer": llm.invoke(state["question"]).content})
graph.add_edge(START, "answer")
graph.add_edge("answer", END)
print(graph.compile().invoke({"question": "Explain observability in one sentence."})["answer"])
