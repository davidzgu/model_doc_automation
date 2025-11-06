from .data_loader_tools import csv_loader


from .tool_registry import REGISTRY

def get_tools_for_role(role: str):
    return REGISTRY.by_role(role)

def get_tools_by_tags(*tags: str):
    return REGISTRY.by_tag(*tags)

def get_tools_by_name(name: str):
    return REGISTRY.get_by_name(name)