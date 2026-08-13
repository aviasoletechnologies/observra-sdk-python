"""LlamaIndex + Hugging Face tracing. Requires llama-index-llms-openai and python-dotenv."""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from llama_index.llms.openai_like import OpenAILike as OpenAI

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = OpenAI(model=os.getenv("HUGGINGFACE_MODEL", "meta-llama/Llama-3.1-8B-Instruct"), api_key=os.getenv("HUGGINGFACE_API_KEY"), api_base="https://router.huggingface.co/v1", is_chat_model=True)
print(llm.complete("Explain observability in one sentence.").text)
