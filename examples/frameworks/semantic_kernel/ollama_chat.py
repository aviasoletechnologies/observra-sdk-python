"""Semantic Kernel + Ollama tracing. Requires semantic-kernel and python-dotenv."""

import asyncio
import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.ollama.services.ollama_chat_completion import (
    OllamaChatCompletion,
)

load_dotenv(Path(__file__).with_name(".env"))

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


kernel = Kernel()
kernel.add_service(
    OllamaChatCompletion(
        ai_model_id="gpt-oss:120b-cloud",
        host="https://ollama.com",
    )
)


async def main() -> None:
    result = await kernel.invoke_prompt("Explain observability in one sentence.")
    print(result)


asyncio.run(main())
