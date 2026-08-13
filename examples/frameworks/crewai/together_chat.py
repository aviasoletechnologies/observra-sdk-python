"""CrewAI + Together tracing. Requires crewai and python-dotenv."""

import os
from pathlib import Path

import observra
from crewai import LLM
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()

llm = LLM(model=os.getenv("TOGETHER_MODEL", "together_ai/meta-llama/Llama-3.1-8B-Instruct-Turbo"), api_key=os.getenv("TOGETHER_API_KEY"), api_base="https://api.together.xyz/v1")
print(llm.call("Explain observability in one sentence."))
