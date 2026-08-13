"""CrewAI + DeepSeek tracing. Requires crewai and python-dotenv."""

import os
from pathlib import Path

import observra
from crewai import LLM
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = LLM(model=os.getenv("DEEPSEEK_MODEL", "deepseek/deepseek-chat"), api_key=os.getenv("DEEPSEEK_API_KEY"), api_base="https://api.deepseek.com/v1")
print(llm.call("Explain observability in one sentence."))
