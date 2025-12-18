import os
from mcp.server.fastmcp import FastMCP

# Note: We don't need to initialize FastMCP here if we are just defining functions 
# to be imported by the main server. But if we want them to be standalone, we can.
# For this refactor, I will define them as standalone functions that can be registered.

def validate_input_file(filepath: str) -> str:
    """
    Validates that the input file exists.
    Returns the absolute path if valid, or an error message starting with 'Error:'.
    """
    if not os.path.exists(filepath):
        return f"Error: File not found: {filepath}"
    return os.path.abspath(filepath)
