"""Together native SDK tracing. Install: pip install together python-dotenv."""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from together import Together

load_dotenv(Path(__file__).with_name(".env"))

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
response = client.chat.completions.create(model=os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.1-8B-Instruct-Turbo"), messages=[{"role": "user", "content": "Explain observability in one sentence."}])
print(response.choices[0].message.content)
