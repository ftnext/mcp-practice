# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "strands-agents[openai,gemini,anthropic]>=1.20.0",
#     "tavily-python>=0.7.17",
# ]
# ///

from typing import Any

from strands import Agent, tool
from strands.models.anthropic import AnthropicModel
from strands.models.gemini import GeminiModel
from strands.models.openai import OpenAIModel
from tavily import TavilyClient


@tool
def search(query: str) -> dict[str, Any]:
    print("[DEBUG] search query:", query)
    tavily = TavilyClient()
    response = tavily.search(query)
    print("[DEBUG] search result:")
    for i, result in enumerate(response["results"], start=1):
        print(f"{i}. [{result['title']}]({result['url']})")
        print(result["content"])
        print()
    print("-" * 40)
    return response


models = [
    OpenAIModel(model_id="gpt-5.2"),
    GeminiModel(model_id="gemini-3-flash-preview"),
    AnthropicModel(model_id="claude-sonnet-4-5", max_tokens=1028),
]

for model in models:
    print(model)
    agent = Agent(model=model, tools=[search])
    agent("JAWS-UG主催のAI Builders Dayはどこで開催される？")
    print()
