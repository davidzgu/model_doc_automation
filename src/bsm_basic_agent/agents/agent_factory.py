from __future__ import annotations
from typing import Iterable, Union
from pathlib import Path

from bsm_basic_agent.prompts.loader import load_prompt

from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
# from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent


def built_graph_agent(
    llm,
    tools: Iterable,
    *,
    prompt: Union[str, Path, None] = None,
    with_memory: bool = True,
    default_system: str = (
        "You are a helpful assistant. Use tools when appropriate"
    )
):
    chat_prompt:str = load_prompt(
        source=prompt,
        default_system=default_system
    )
    checkpointer = MemorySaver() if with_memory else None
    agent = create_agent(
        llm,
        list(tools),
        system_prompt=chat_prompt,
        checkpointer = checkpointer
    )
    return agent