"""LlamaIndex + Groq with Observra tracing.

Requires python-dotenv, llama-index-core, and llama-index-llms-groq.
Values load from sibling .env.
"""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from llama_index.llms.groq import Groq

load_dotenv(Path(__file__).with_name(".env"))

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = Groq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
response = llm.complete("Explain observability in one sentence.")
print(response.text)
