"""Semantic Kernel + Anthropic tracing. Requires semantic-kernel and python-dotenv."""

import asyncio
import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.anthropic.services.anthropic_chat_completion import (
    AnthropicChatCompletion,
)

load_dotenv(Path(__file__).with_name(".env"))

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


kernel = Kernel()
kernel.add_service(
    AnthropicChatCompletion(
        ai_model_id="claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
)


async def main() -> None:
    result = await kernel.invoke_prompt("Explain observability in one sentence.")
    print(result)


asyncio.run(main())
