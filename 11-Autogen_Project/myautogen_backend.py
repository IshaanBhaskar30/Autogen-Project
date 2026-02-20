"""
Multi-agent literature-review assistant
AutoGen 0.4+ + Groq (OpenAI-compatible endpoint)
"""

from __future__ import annotations
import asyncio
from typing import AsyncGenerator, Dict, List

import arxiv
from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient



# 1. arXiv Tool


def arxiv_search(query: str, max_results: int = 5) -> List[Dict]:
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers: List[Dict] = []
    for result in client.results(search):
        papers.append(
            {
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "published": result.published.strftime("%Y-%m-%d"),
                "summary": result.summary,
                "pdf_url": result.pdf_url,
            }
        )
    return papers


arxiv_tool = FunctionTool(
    arxiv_search,
    description="Searches arXiv and returns relevant papers.",
)



# 2. Build Team (Groq via OpenAI-compatible endpoint)


def build_team(api_key: str) -> RoundRobinGroupChat:
    if not api_key:
        raise ValueError("Groq API key is required.")

    llm_client = OpenAIChatCompletionClient(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        model_info={
            "family": "llama3",
            "vision": False,
            "function_calling": True,
            "json_output": True,
        },
        temperature=0.2,
        max_tokens=2048,
    )

    search_agent = AssistantAgent(
        name="search_agent",
        description="Crafts arXiv queries and retrieves candidate papers.",
        system_message=(
            "Given a user topic, think of the best arXiv query and call the "
            "provided tool. Fetch five-times the requested number, then "
            "select exactly the requested number and pass them as JSON."
        ),
        tools=[arxiv_tool],
        model_client=llm_client,
        reflect_on_tool_use=False,
    )

    summarizer = AssistantAgent(
        name="summarizer",
        description="Produces a Markdown literature review.",
        system_message=(
            "You are an expert researcher. When given paper JSON, write:\n"
            "1. 2–3 sentence introduction\n"
            "2. One bullet per paper (title as link, authors, problem, contribution)\n"
            "3. One-sentence takeaway"
        ),
        model_client=llm_client,
    )

    return RoundRobinGroupChat(
        participants=[search_agent, summarizer],
        max_turns=2,
    )



# 3. Orchestrator


async def run_litrev(
    topic: str,
    num_papers: int,
    api_key: str,
) -> AsyncGenerator[str, None]:

    team = build_team(api_key)

    task_prompt = (
        f"Conduct a literature review on **{topic}** "
        f"and return exactly {num_papers} papers."
    )

    async for msg in team.run_stream(task=task_prompt):
        if isinstance(msg, TextMessage):
            yield f"{msg.source}: {msg.content}"



# 4. CLI Test


if __name__ == "__main__":
    async def _demo():
        key = input("Enter your Groq API key: ").strip()
        async for line in run_litrev(
            "Artificial Intelligence",
            5,
            key,
        ):
            print(line)

    asyncio.run(_demo())