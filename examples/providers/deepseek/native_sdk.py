"""DeepSeek OpenAI-compatible SDK tracing. Install: pip install openai python-dotenv."""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1")
response = client.chat.completions.create(model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"), messages=[{"role": "user", "content": "Explain observability in one sentence."}])
print(response.choices[0].message.content)
