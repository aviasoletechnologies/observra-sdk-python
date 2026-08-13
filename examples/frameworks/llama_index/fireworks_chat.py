"""LlamaIndex + Fireworks tracing. Requires llama-index-llms-openai and python-dotenv."""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from llama_index.llms.openai_like import OpenAILike as OpenAI

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = OpenAI(model=os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p1-8b-instruct"), api_key=os.getenv("FIREWORKS_API_KEY"), api_base="https://api.fireworks.ai/inference/v1", is_chat_model=True)
print(llm.complete("Explain observability in one sentence.").text)
