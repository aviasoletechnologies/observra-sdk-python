"""Bounded LlamaIndex supervisor/worker workflow + Groq tracing.

Requires python-dotenv, llama-index-core, and llama-index-llms-groq.
Values load from sibling .env.
"""

import asyncio
import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.groq import Groq

load_dotenv(Path(__file__).with_name(".env"))

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()



def research_observability(question: str) -> str:
    """Return concise research answer about observability."""
    return (
        "Observability lets teams understand system behavior from telemetry, "
        "detect failures quickly, and diagnose root causes with less downtime."
    )


async def main() -> None:
    llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
    research_tool = FunctionTool.from_defaults(
        fn=research_observability,
        return_direct=True,
    )
    researcher = FunctionAgent(
        name="researcher",
        description="Researches observability questions and returns concise factual answers.",
        system_prompt=(
            "Call research_observability once for research questions. Do not hand off work."
        ),
        tools=[research_tool],
        can_handoff_to=[],
        llm=llm,
        streaming=False,
        early_stopping_method="force",
    )
    supervisor = FunctionAgent(
        name="supervisor",
        description="Delegates research to researcher, then returns the result.",
        system_prompt="Hand off to researcher once for research questions. Do not repeat handoffs.",
        can_handoff_to=["researcher"],
        llm=llm,
        streaming=False,
        early_stopping_method="force",
    )
    workflow = AgentWorkflow(
        agents=[supervisor, researcher],
        root_agent="supervisor",
        timeout=30,
        early_stopping_method="force",
    )
    result = await workflow.run(
        user_msg="Why does observability matter?",
        max_iterations=3,
    )
    print(str(result))


if __name__ == "__main__":
    asyncio.run(main())
