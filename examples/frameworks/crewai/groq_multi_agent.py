"""CrewAI bounded sequential multi-agent Crew + Groq tracing.

Requires crewai and python-dotenv. Values load from sibling .env.
"""

import os
from pathlib import Path

import observra
from crewai import Agent, Crew, LLM, Process, Task
from crewai.hooks import before_llm_call
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()



@before_llm_call
def remove_groq_unsupported_cache_breakpoints(context):
    """Remove CrewAI prompt-cache metadata unsupported by Groq."""
    for message in context.messages:
        message.pop("cache_breakpoint", None)


llm = LLM(model="groq/llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
researcher = Agent(
    role="Researcher",
    goal="Explain observability accurately in one sentence.",
    backstory="You give concise factual explanations.",
    llm=llm,
    max_iter=1,
)
editor = Agent(
    role="Editor",
    goal="Return final answer unchanged when already concise and factual.",
    backstory="You preserve useful technical wording.",
    llm=llm,
    max_iter=1,
)
research_task = Task(
    description="Explain why observability matters in one sentence.",
    expected_output="One concise factual sentence.",
    agent=researcher,
)
edit_task = Task(
    description="Return previous task's answer as final answer without adding facts.",
    expected_output="One concise factual sentence.",
    agent=editor,
    context=[research_task],
)
crew = Crew(
    agents=[researcher, editor],
    tasks=[research_task, edit_task],
    process=Process.sequential,
)
print(crew.kickoff())
