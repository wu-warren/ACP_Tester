"""Prompt the ReAct agent to join and play an ACP game session.

Run from the Autonomous-Reasoning-Agent repo root:
    python scripts/acp_play_game.py <session_id>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.acp_react import run_tool_only_react


def build_prompt(session_id: str) -> str:
    return f"read skill.md and play altruAgent session_id {session_id}."


def configure_profile(profile: str | None, env_file: str | None) -> None:
    if env_file:
        os.environ["ACP_LOCAL_ENV"] = env_file
    elif profile:
        os.environ["ACP_LOCAL_ENV"] = f".env.{profile}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Join and play an ACP game session.")
    parser.add_argument("session_id")
    parser.add_argument("--profile", help="Use .env.<profile> for this agent's ACP API_KEY.")
    parser.add_argument("--env-file", help="Use this env file for this agent's ACP API_KEY.")
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    configure_profile(args.profile, args.env_file)

    result = run_tool_only_react(
        build_prompt(args.session_id),
        max_iterations=args.max_iterations,
        model=args.model,
    )
    print(f"\nCompleted after {result.iterations} iterations.")


if __name__ == "__main__":
    main()
