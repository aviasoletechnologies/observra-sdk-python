"""LangChain bounded supervisor/worker multi-agent + Groq traces.

Requires python-dotenv, langchain, langchain-core, and langchain-groq. Values
load from sibling .env.
"""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq

load_dotenv(Path(__file__).with_name(".env"))

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()



def last_message_text(result: dict) -> str:
    return str(result["messages"][-1].content)


llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
researcher = create_agent(
    llm,
    tools=[],
    system_prompt="You are a research worker. Give concise factual answers.",
)


@tool(return_direct=True)
def ask_researcher(question: str) -> str:
    """Delegate one research question to a specialized worker agent."""
    result = researcher.invoke(
        {"messages": [{"role": "user", "content": question}]},
        {"recursion_limit": 2},
    )
    return last_message_text(result)


supervisor = create_agent(
    llm,
    tools=[ask_researcher],
    system_prompt=(
        "Call ask_researcher exactly once, then return its result directly. "
        "Do not call any tool again."
    ),
)
result = supervisor.invoke(
    {"messages": [{"role": "user", "content": "Ask researcher why observability matters."}]},
    {"recursion_limit": 3},
)
print(last_message_text(result))
