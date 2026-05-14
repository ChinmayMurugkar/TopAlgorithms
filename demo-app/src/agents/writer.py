"""
writer agent — Part of demo-app (Built with AgentVoy)
"""

from agents import Agent, Runner
from src.tools.tools import get_tools


def create_agent() -> Agent:
    """Create and configure the agent with AgentVoy guardrails."""
    tools = get_tools()

    agent = Agent(
        name="writer",
        instructions="""You are a helpful AI assistant.

Follow these guidelines:
- Be concise and accurate
- Ask for clarification when the request is ambiguous
- Respect the guardrails defined in agent.guard.yml
""",
        model="gpt-4o",
        tools=tools,
    )

    return agent


def run_agent(prompt: str) -> str:
    """Run the agent with the given prompt, enforcing agent.guard.yml at runtime."""
    import asyncio
    from agentvoy_guard import Guard
    guard = Guard.from_config()

    async def _run():
        agent = create_agent()
        result = await Runner.run(
            agent,
            prompt,
            max_turns=20,
        )
        return result.final_output or ""

    with guard.session() as session:
        session.check_input(prompt)
        final = asyncio.run(_run())
        session.check_output(final)

    print(f"[guard] {guard.last_summary}")
    return final
