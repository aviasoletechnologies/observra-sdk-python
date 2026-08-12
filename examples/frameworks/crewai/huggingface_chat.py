"""CrewAI + Hugging Face tracing. Requires crewai and python-dotenv."""

import os
from pathlib import Path

import observra
from crewai import LLM
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = LLM(model=os.getenv("HUGGINGFACE_MODEL", "huggingface/meta-llama/Llama-3.1-8B-Instruct"), api_key=os.getenv("HUGGINGFACE_API_KEY"), api_base="https://router.huggingface.co/v1")
print(llm.call("Explain observability in one sentence."))
