"""Hugging Face native SDK tracing. Install: pip install huggingface-hub python-dotenv."""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv(Path(__file__).with_name(".env"))

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


client = InferenceClient(api_key=os.getenv("HUGGINGFACE_API_KEY"))
response = client.chat_completion(model=os.getenv("HUGGINGFACE_MODEL", "meta-llama/Llama-3.1-8B-Instruct"), messages=[{"role": "user", "content": "Explain observability in one sentence."}])
print(response.choices[0].message.content)
