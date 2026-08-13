"""CrewAI normal LLM call + OpenRouter tracing.

Requires crewai and python-dotenv. Values load from sibling .env.
"""

import os
from pathlib import Path

import observra
from crewai import LLM
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = LLM(
    model=os.getenv("OPENROUTER_MODEL", "openrouter/cohere/north-mini-code:free"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
print(llm.call("Explain observability in one sentence."))
