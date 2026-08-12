# observra

Zero-touch tracing and gateway routing for LLM SDKs — configure once, keep writing the provider SDK's own client exactly as before.

## Install

```bash
pip install observra
```


## Configure

Only `gateway_key` is required — `gateway_url` defaults to the production gateway (`https://gateway.observra.in`):

```python
import observra

observra.configure(gateway_key="obs_live_xxx")
```

Or via environment variable (no explicit `configure()` call needed):

```bash
export OBSERVRA_GATEWAY_KEY="obs_live_xxx"
```

## Use

No wrapper client — write plain provider SDK code exactly as you already would. `configure()` transparently patches the SDK (Gemini) and, for every known provider host, patches `httpx` itself (OpenAI, Anthropic — and Gemini too, for non-SDK callers), so any request to those hosts routes through the gateway and gets traced/guardrailed, no matter which library actually made the call:

```python
from google import genai

client = genai.Client(api_key="AIza...")  # your own Gemini key, forwarded as-is

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Summarize this support ticket: ...",
)
print(response.text)
```

```python
from openai import OpenAI

client = OpenAI(api_key="sk-...")  # your own OpenAI key
response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "..."}])
```

```python
from anthropic import Anthropic

client = Anthropic(api_key="sk-ant-...")  # your own Anthropic key
response = client.messages.create(model="claude-3-5-sonnet-20241022", max_tokens=1024, messages=[{"role": "user", "content": "..."}])
```

Same for async clients, and for any framework integration that builds one of these internally — e.g. LangChain's `ChatGoogleGenerativeAI` routes through the gateway automatically too, no extra step. See `examples/raw_http_gemini.py` for the same guarantee at the raw-HTTP level — no provider SDK at all, just `httpx` pointed straight at Google's real endpoint.

Inside a LangChain agent — `instrument()` additionally traces every step (agent/chain/tool boundaries) under one trace, on top of the LLM-call-level tracing `configure()` already gives you:

```python
observra.instrument()
```

Guardrails (PII/secret detection on prompts and responses) run on every call automatically — violations are recorded as span events (`guardrail.violation`), the payload itself is never blocked or altered.

See [`examples/`](examples/) for full runnable scripts, and traces show up in your Observra dashboard's Request Flow view.

