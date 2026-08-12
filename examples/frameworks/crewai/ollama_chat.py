"""CrewAI normal LLM call + Ollama tracing.

Requires crewai and python-dotenv. Values load from sibling .env.
"""

import os
from pathlib import Path

import observra
from crewai import LLM
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = LLM(
    model=f"ollama_chat/gpt-oss:120b-cloud",
    api_base=os.getenv("OLLAMA_BASE_URL", "https://ollama.com"),
    api_key=os.getenv("OLLAMA_API_KEY"),
)
print(llm.call("Explain observability in one sentence."))
