from __future__ import annotations
from typing import Union, Iterable
from pathlib import Path


def load_prompt(
    source: Union[str, Path, None] = None,
    default_system: str = "You are a helpful assistant. Use tools when appropriate"
) -> str:
    if source is None:
        return default_system
    
    p = Path(str(source))
    if p.exists() and p.is_file():
        txt = p.read_text(encoding='utf-8')
        return txt
    
    return (str(source))