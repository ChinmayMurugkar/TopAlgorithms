"""
test-anthropic-sdk — Built with AgentVoy
https://github.com/agentvoy
"""

import anthropic
from tools import get_tools, process_tool_call


def create_client() -> anthropic.Anthropic:
    """Create the Anthropic client."""
    return anthropic.Anthropic()


def run_agent(prompt: str) -> str:
    """Run the agent with an agentic loop."""
    client = create_client()
    tools = get_tools()
    messages = [{"role": "user", "content": prompt}]

    iteration = 0
    max_iterations = 20

    while iteration < max_iterations:
        iteration += 1

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8096,
            tools=tools,
            messages=messages,
        )

        # Add assistant response to messages
        messages.append({"role": "assistant", "content": response.content})

        # If no tool calls, we're done
        if response.stop_reason == "end_turn":
            # Extract final text response
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return "Done."

        # Process tool calls
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = process_tool_call(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    })

            messages.append({"role": "user", "content": tool_results})

    return "Max iterations reached."
