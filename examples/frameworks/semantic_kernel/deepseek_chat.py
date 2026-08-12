"""Semantic Kernel + DeepSeek tracing. Requires semantic-kernel and python-dotenv."""

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

load_dotenv(Path(__file__).with_name(".env"))

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


kernel = Kernel()
kernel.add_service(OpenAIChatCompletion(ai_model_id=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"), async_client=AsyncOpenAI(api_key=os.getenv("DEEPSEEK_API_KEY") or "", base_url="https://api.deepseek.com/v1")))


async def main() -> None:
    print(await kernel.invoke_prompt("Explain observability in one sentence."))


asyncio.run(main())
