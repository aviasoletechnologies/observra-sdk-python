"""LangChain one-tool agent + Groq with bounded Observra traces.

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



@tool(return_direct=True)
def refund_policy() -> str:
    """Return company refund policy."""
    return "Refunds are processed within 5-7 business days."


llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
agent = create_agent(
    llm,
    tools=[refund_policy],
    system_prompt=(
        "Call refund_policy once for refund-policy questions. "
        "Its result is final; do not make further tool calls."
    ),
)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is our refund policy?"}]},
    {"recursion_limit": 3},
)
print(result["messages"][-1].content)
