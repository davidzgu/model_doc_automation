# -*- coding: utf-8 -*-
from langchain_ollama import ChatOllama

# def get_llm():
#     llm = ChatOllama(
#         model="qwen2.5:7b",
#         temperature=0,
#     )
#     return llm

def get_llm():
    llm = ChatOllama(
        model="qwen2.5:32b",
        temperature=0,
    )
    return llm