"""Raw HTTP call to Google's real Gemini endpoint — no google-genai client, no gateway-specific code at all.

This is exactly what calling Gemini directly looks like, with one added
line: observra.configure(). That's enough — observra patches httpx itself,
so any request to a known provider host (generativelanguage.googleapis.com
here) gets silently rerouted through the gateway, traced, and guardrailed.
Compare with basic_gemini.py (google-genai Client, also auto-patched) and
langchain_agent.py (framework-level, also auto-patched).
"""

import os
from pathlib import Path

import httpx
import observra
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()

MODEL = "gemini-3.1-flash-lite"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
payload = {
    "contents": [
        {
            "role": "user",
            "parts": [{"text": "Summarize this support ticket: the customer can't reset their password."}],
        }
    ]
}

response = httpx.post(url, headers={"x-goog-api-key": GEMINI_API_KEY}, json=payload, timeout=60)
response.raise_for_status()

print(response.json()["candidates"][0]["content"]["parts"][0]["text"])
