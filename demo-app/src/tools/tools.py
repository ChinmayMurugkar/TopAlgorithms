"""
Agent tools — add your custom tools here.
"""

from agents import function_tool


@function_tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # TODO: Implement your search logic
    return f"Search results for: {query}"


@function_tool
def read_file(path: str) -> str:
    """Read the contents of a file."""
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"File not found: {path}"
    except PermissionError:
        return f"Permission denied: {path}"


def get_tools() -> list:
    """Return all available tools."""
    return [search_web, read_file]
