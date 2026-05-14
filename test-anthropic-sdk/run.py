"""
Run the test-anthropic-sdk agent.
"""

from dotenv import load_dotenv
from agent import run_agent

load_dotenv()


def main():
    print("\n🚀 test-anthropic-sdk — Powered by AgentVoy")
    print("=" * 50)
    print("Type your prompt (or 'quit' to exit):\n")

    while True:
        try:
            prompt = input("> ")
            if prompt.lower() in ("quit", "exit", "q"):
                print("\nGoodbye!")
                break
            if not prompt.strip():
                continue

            print("\nThinking...\n")
            result = run_agent(prompt)
            print(f"\n{result}\n")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    main()
