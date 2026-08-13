"""LangChain + Together tracing. Requires langchain-openai and python-dotenv."""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = ChatOpenAI(model=os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.1-8B-Instruct-Turbo"), api_key=os.getenv("TOGETHER_API_KEY"), base_url="https://api.together.xyz/v1")
print(llm.invoke("Explain observability in one sentence.").content)
