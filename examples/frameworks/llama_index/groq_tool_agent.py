"""Bounded LlamaIndex FunctionAgent + Groq with Observra tracing.

Requires python-dotenv, llama-index-core, and llama-index-llms-groq.
Values load from sibling .env.
"""

import asyncio
import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.groq import Groq

load_dotenv(Path(__file__).with_name(".env"))

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()



def refund_policy() -> str:
    """Return company refund policy."""
    return "Refunds are processed within 5-7 business days."


async def main() -> None:
    refund_tool = FunctionTool.from_defaults(fn=refund_policy, return_direct=True)
    agent = FunctionAgent(
        name="RefundAgent",
        description="Answers refund-policy questions.",
        system_prompt="Call refund_policy once for refund-policy questions.",
        tools=[refund_tool],
        llm=Groq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY")),
        streaming=False,
        early_stopping_method="force",
        timeout=20,
    )
    result = await agent.run(user_msg="What is our refund policy?", max_iterations=3)
    print(str(result))


if __name__ == "__main__":
    asyncio.run(main())
