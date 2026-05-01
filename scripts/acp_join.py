"""Prompt the ReAct agent to register with ACP and save its API key.

Run from the Autonomous-Reasoning-Agent repo root:
    python scripts/acp_join.py --name MyAgent
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.acp_react import run_tool_only_react
def build_prompt(name: str, description: str) -> str:
    return "read skill.md and join altruAgent."


def main() -> None:
    parser = argparse.ArgumentParser(description="Register an ACP agent and save API_KEY to .env.")
    parser.add_argument("--name", default="AutonomousReasoningAgent")
    parser.add_argument("--description", default="ReAct ACP player")
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    result = run_tool_only_react(
        build_prompt(args.name, args.description),
        max_iterations=args.max_iterations,
        model=args.model,
    )
    print(f"\nCompleted after {result.iterations} iterations.")


if __name__ == "__main__":
    main()
