"""
Agent tools — add your custom tools here.
"""


def get_tools() -> list:
    """Return tool definitions for the Anthropic API."""
    return [
        {
            "name": "search_web",
            "description": "Search the web for information on a given topic.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    }
                },
                "required": ["query"],
            },
        },
        {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read.",
                    }
                },
                "required": ["path"],
            },
        },
    ]


def process_tool_call(tool_name: str, tool_input: dict) -> str:
    """Execute a tool call and return the result."""
    if tool_name == "search_web":
        return _search_web(tool_input["query"])
    elif tool_name == "read_file":
        return _read_file(tool_input["path"])
    else:
        return f"Unknown tool: {tool_name}"


def _search_web(query: str) -> str:
    """Search the web for information."""
    # TODO: Implement your search logic (e.g., Tavily, Serper, Brave Search)
    return f"Search results for: {query}"


def _read_file(path: str) -> str:
    """Read a file's contents."""
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"File not found: {path}"
    except PermissionError:
        return f"Permission denied: {path}"
