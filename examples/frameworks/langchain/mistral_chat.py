"""LangChain + Mistral tracing. Requires langchain-openai and python-dotenv."""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).with_name(".env"))

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = ChatOpenAI(model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"), api_key=os.getenv("MISTRAL_API_KEY"), base_url="https://api.mistral.ai/v1")
print(llm.invoke("Explain observability in one sentence.").content)
