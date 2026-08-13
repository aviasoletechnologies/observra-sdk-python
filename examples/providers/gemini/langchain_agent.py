"""observra used inside a LangChain agent — zero-touch tracing and gateway routing.

observra.configure() patches google.genai.Client itself (see basic_gemini.py),
so ChatGoogleGenerativeAI — which builds its own genai.Client internally —
routes through the gateway automatically, no client_args needed.
observra.instrument() separately patches LangChain's callback manager so
each agent/chain/tool step becomes its own span nested under one trace.

Requires langchain>=1.0 (langgraph-based agents — `AgentExecutor` /
`langchain.agents.create_react_agent` were removed in LangChain 1.0 in favor
of `langgraph.prebuilt.create_react_agent`).
"""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument() 

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)


@tool
def search_docs(query: str) -> str:
    """Search internal docs."""
    return "Refunds are processed within 5-7 business days."


agent = create_agent(llm, tools=[search_docs])

result = agent.invoke({"messages": [{"role": "user", "content": "What's our refund policy?"}]})
print(result["messages"][-1].content)
