from __future__ import annotations
from typing import Union, Iterable
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

Message = tuple[str, str]

def _from_text(system_text: str)-> ChatPromptTemplate:
    msgs: list[Message] = [
        ("system", system_text),
        ("placeholder", "{message}")
    ]
    return ChatPromptTemplate.from_messages(msgs)


def load_prompt(
    source: Union[str, Path, ChatPromptTemplate, None],
    default_system: str = "You are a helpful assistant. Use tools when appropriate"
) -> ChatPromptTemplate:
    if source is None:
        return _from_text(default_system)
    
    if isinstance(source, ChatPromptTemplate):
        return source
    
    p = Path(str(source))
    if p.exists() and p.is_file():
        txt = p.read_text(encoding='utf-8')
        return _from_text(txt)
    
    return _from_text(str(source))