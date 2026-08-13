"""LangChain + Fireworks tracing. Requires langchain-openai and python-dotenv."""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = ChatOpenAI(model=os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p1-8b-instruct"), api_key=os.getenv("FIREWORKS_API_KEY"), base_url="https://api.fireworks.ai/inference/v1")
print(llm.invoke("Explain observability in one sentence.").content)
