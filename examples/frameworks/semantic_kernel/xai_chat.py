"""Semantic Kernel + xAI tracing. Requires semantic-kernel and python-dotenv."""

import asyncio
import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from openai import AsyncOpenAI
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import (
    OpenAIChatCompletion,
)

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


kernel = Kernel()
kernel.add_service(OpenAIChatCompletion(ai_model_id=os.getenv("XAI_MODEL", "grok-3-mini"), async_client=AsyncOpenAI(api_key=os.getenv("XAI_API_KEY") or "", base_url="https://api.x.ai/v1")))


async def main() -> None:
    print(await kernel.invoke_prompt("Explain observability in one sentence."))


asyncio.run(main())
