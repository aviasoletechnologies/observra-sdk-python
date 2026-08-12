"""LlamaIndex + OpenAI with Observra tracing.

Requires python-dotenv, llama-index-core, and llama-index-llms-openai.
Values load from sibling .env.
"""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI

load_dotenv(Path(__file__).with_name(".env"))

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
response = llm.complete("Explain observability in one sentence.")
print(response.text)
