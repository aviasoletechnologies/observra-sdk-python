"""LangChain + Cohere tracing. Requires langchain-openai and python-dotenv."""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = ChatOpenAI(model=os.getenv("COHERE_MODEL", "command-a-03-2025"), api_key=os.getenv("COHERE_API_KEY"), base_url="https://api.cohere.com/compatibility/v1")
print(llm.invoke("Explain observability in one sentence.").content)
