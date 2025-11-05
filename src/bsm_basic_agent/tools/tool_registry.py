from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Iterable
from langchain_core.tools import BaseTool

@dataclass
class ToolMeta:
    name: str
    tool: BaseTool
    tags: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)

class ToolRegistry:
    def __init__(self) -> None:
        self._by_name: Dict[str, ToolMeta] = {}

    def register(
        self, tool:BaseTool, *, 
        name: Optional[str]=None,
        tags: Iterable[str]=(),
        roles: Iterable[str]=(),
    ) -> BaseTool:
        n = name or tool.name
        meta = ToolMeta(
            name=n, 
            tool=tool,
            tags=list(tags),
            roles=list(roles)
        )
        self._by_name[n] = meta
        return tool
    
    def get_by_name(self, name:str) -> List[BaseTool]:
        return self._by_name[name].tool
    
    def list(self) -> List[BaseTool]:
        return [m.tool for m in self._by_name.values()]
    
    def by_tag(self, *tags: str) -> List[BaseTool]:
        ts = set(tags)
        return [m.tool for m in self._by_name.values() if ts.issubset(set(m.tags))]

    def by_role(self, *roles: str) -> List[BaseTool]:
        rs = set(roles)
        return [m.tool for m in self._by_name.values() if rs.issubset(set(m.roles))]
    
REGISTRY = ToolRegistry()

def register_tool(
    *, 
    name: Optional[str]=None,
    tags: Iterable[str]=(),
    roles: Iterable[str]=()
):
    def deco(t:BaseTool) -> BaseTool:
        REGISTRY.register(
            tool=t,
            name=name,
            tags=tags,
            roles=roles
        )
        return t
    return deco

