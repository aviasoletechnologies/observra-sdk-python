"""Semantic Kernel + Gemini tracing. Requires semantic-kernel and python-dotenv."""

import asyncio
import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.google.google_ai.services.google_ai_chat_completion import (
    GoogleAIChatCompletion,
)

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


kernel = Kernel()
kernel.add_service(
    GoogleAIChatCompletion(
        gemini_model_id="gemini-3.1-flash-lite",
        api_key=os.getenv("GEMINI_API_KEY"),
    )
)


async def main() -> None:
    result = await kernel.invoke_prompt("Explain observability in one sentence.")
    print(result)


asyncio.run(main())
