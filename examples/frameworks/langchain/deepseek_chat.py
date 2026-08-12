"""LangChain + DeepSeek tracing. Requires langchain-openai and python-dotenv."""

import os
from pathlib import Path

import observra
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).with_name(".env"))

observra.configure(gateway_key=os.getenv("GATEWAY_KEY"))
observra.instrument()


llm = ChatOpenAI(model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"), api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1")
print(llm.invoke("Explain observability in one sentence.").content)
