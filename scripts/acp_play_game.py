"""Prompt the ReAct agent to join and play an ACP game session.

Run from the Autonomous-Reasoning-Agent repo root:
    python scripts/acp_play_game.py <session_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.acp_react import run_tool_only_react
def build_prompt(session_id: str) -> str:
    return f"read skill.md and play altruAgent session_id {session_id}."


def main() -> None:
    parser = argparse.ArgumentParser(description="Join and play an ACP game session.")
    parser.add_argument("session_id")
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    result = run_tool_only_react(
        build_prompt(args.session_id),
        max_iterations=args.max_iterations,
        model=args.model,
    )
    print(f"\nCompleted after {result.iterations} iterations.")


if __name__ == "__main__":
    main()
