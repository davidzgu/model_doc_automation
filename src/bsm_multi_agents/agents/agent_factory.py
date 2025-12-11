from __future__ import annotations
from typing import Iterable, Union
from pathlib import Path

from bsm_basic_agent.prompts.loader import load_prompt
from bsm_multi_agents.tools import get_tools_for_role
from bsm_multi_agents.config.llm_config_claude import get_llm
from bsm_multi_agents.graph.state import WorkflowState

from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
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
    system_prompt: str = DEFAULTSYSTEM
):
    chat_prompt:str = load_prompt(
        source=prompt,
        default_system=system_prompt
    )
    # checkpointer = MemorySaver() if with_memory else None
    agent = create_react_agent(
        llm,
        list(tools),
        state_schema=WorkflowState,   
        prompt=chat_prompt,
        # checkpointer = checkpointer,
    )
    return agent


def built_graph_agent_by_role(
    agent_role: str,
    system_prompt: str = DEFAULTSYSTEM
):
    tools = get_tools_for_role(agent_role)
    llm = get_llm()

    # Display agent configuration
    llm_model = getattr(llm, 'model_name', None) or getattr(llm, 'model', 'unknown')
    llm_provider = llm.__class__.__name__
    tool_names = [t.name for t in tools]

    print(f"[{agent_role}] LLM: {llm_provider} ({llm_model}), Tools: {tool_names}")

    agent = built_graph_agent(llm, tools, system_prompt=system_prompt)
    return agent