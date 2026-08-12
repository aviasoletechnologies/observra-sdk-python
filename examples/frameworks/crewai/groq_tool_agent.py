"""CrewAI bounded tool agent + Groq tracing.

Requires crewai and python-dotenv. Values load from sibling .env.
"""

import os
from pathlib import Path

import observra
from crewai import Agent, Crew, LLM, Task
from crewai.hooks import before_llm_call
from crewai.tools import tool
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()



@before_llm_call
def remove_groq_unsupported_cache_breakpoints(context):
    """Remove CrewAI prompt-cache metadata unsupported by Groq."""
    for message in context.messages:
        message.pop("cache_breakpoint", None)


@tool("observability_fact")
def observability_fact(topic: str) -> str:
    """Return one fact about supplied observability topic."""
    return f"Observability for {topic} uses traces, metrics, and logs to explain system behavior."


llm = LLM(model="groq/llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
agent = Agent(
    role="Observability researcher",
    goal="Answer with one verified fact.",
    backstory="You use supplied tools and keep answers concise.",
    tools=[observability_fact],
    llm=llm,
    max_iter=2,
)
task = Task(
    description=(
        "Call observability_fact exactly once with topic='observability'. "
        "Return its result without additions."
    ),
    expected_output="One factual sentence from observability_fact.",
    agent=agent,
)
print(Crew(agents=[agent], tasks=[task]).kickoff())
