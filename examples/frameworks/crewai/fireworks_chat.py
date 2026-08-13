"""CrewAI + Fireworks tracing. Requires crewai and python-dotenv."""

import os
from pathlib import Path

import observra
from crewai import LLM
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = LLM(model=os.getenv("FIREWORKS_MODEL", "fireworks_ai/accounts/fireworks/models/llama-v3p1-8b-instruct"), api_key=os.getenv("FIREWORKS_API_KEY"), api_base="https://api.fireworks.ai/inference/v1")
print(llm.call("Explain observability in one sentence."))
