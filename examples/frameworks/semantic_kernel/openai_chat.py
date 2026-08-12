"""Semantic Kernel + OpenAI tracing. Requires semantic-kernel and python-dotenv."""

import asyncio
import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import (
    OpenAIChatCompletion,
)

load_dotenv(Path(__file__).with_name(".env"))

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


kernel = Kernel()
kernel.add_service(
    OpenAIChatCompletion(
        ai_model_id="gpt-4.1-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
)


async def main() -> None:
    result = await kernel.invoke_prompt("Explain observability in one sentence.")
    print(result)


asyncio.run(main())
