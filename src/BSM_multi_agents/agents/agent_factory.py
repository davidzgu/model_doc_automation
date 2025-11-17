from __future__ import annotations
from typing import Iterable, Union
from pathlib import Path

from bsm_basic_agent.prompts.loader import load_prompt
from bsm_multi_agents.tools import get_tools_for_role
from bsm_multi_agents.config.llm_config import get_llm

from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
# from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent


DEFAULTSYSTEM = (
    "You are a helpful assistant. Use tools when appropriate"
)

def built_graph_agent(
    llm,
    tools: Iterable,
    *,
    prompt: Union[str, Path, None] = None,
    with_memory: bool = True,
    default_system: str = DEFAULTSYSTEM
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


def built_graph_agent_by_role(
    agent_role: str,
    default_system: str = DEFAULTSYSTEM
):
    tools = get_tools_for_role(agent_role)
    llm = get_llm()
    agent = built_graph_agent(llm, tools, default_system=default_system)
    return agent