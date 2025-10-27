# -*- coding: utf-8 -*-
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate

def build_agent(llm, tools):
    # Create a prompt template with system message
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful quant assistant. Use tools when appropriate."),
        ("placeholder", "{messages}"),
    ])

    memory = MemorySaver()
    agent = create_react_agent(llm, tools, prompt=prompt, checkpointer=memory)
    return agent