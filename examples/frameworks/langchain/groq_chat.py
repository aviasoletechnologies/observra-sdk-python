"""LangChain + Groq with Observra tracing.

Requires python-dotenv and langchain-groq. Values load from sibling .env.
"""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
response = llm.invoke("Explain observability in one sentence.")
print(response.content)
