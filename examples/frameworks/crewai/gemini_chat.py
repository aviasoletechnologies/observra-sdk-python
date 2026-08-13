"""CrewAI normal LLM call + Gemini tracing.

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


llm = LLM(model="gemini/gemini-3.1-flash-lite", api_key=os.getenv("GEMINI_API_KEY"))
print(llm.call("Explain observability in one sentence."))
