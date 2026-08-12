"""CrewAI + Cohere tracing. Requires crewai and python-dotenv."""

import os
from pathlib import Path

import observra
from crewai import LLM
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = LLM(model=os.getenv("COHERE_MODEL", "cohere/command-a-03-2025"), api_key=os.getenv("COHERE_API_KEY"), api_base="https://api.cohere.com/compatibility/v1")
print(llm.call("Explain observability in one sentence."))
