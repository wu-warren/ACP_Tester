"""Prompt the ReAct agent to register with ACP and save its API key.

Run from the Autonomous-Reasoning-Agent repo root:
    python scripts/acp_join.py --name MyAgent
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


def build_prompt(name: str, description: str) -> str:
    return "read skill.md and join altruAgent."


def configure_profile(profile: str | None, env_file: str | None) -> None:
    if env_file:
        os.environ["ACP_LOCAL_ENV"] = env_file
    elif profile:
        os.environ["ACP_LOCAL_ENV"] = f".env.{profile}"


def configure_backend(local_backend: bool, backend_url: str | None) -> None:
    if backend_url:
        os.environ["ACP_BACKEND_URL"] = backend_url.rstrip("/")
    elif local_backend:
        os.environ["ACP_BACKEND_URL"] = "http://localhost:3000"


def main() -> None:
    parser = argparse.ArgumentParser(description="Register an ACP agent and save API_KEY to .env.")
    parser.add_argument("--name", default="AutonomousReasoningAgent")
    parser.add_argument("--description", default="ReAct ACP player")
    parser.add_argument("--profile", help="Use .env.<profile> for this agent's ACP API_KEY.")
    parser.add_argument("--env-file", help="Use this env file for this agent's ACP API_KEY.")
    parser.add_argument("--local-backend", action="store_true", help="Use http://localhost:3000 as the control plane.")
    parser.add_argument("--backend-url", help="Use this control-plane base URL instead of Agent_ACP/cdk/.env.")
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    configure_profile(args.profile, args.env_file)
    configure_backend(args.local_backend, args.backend_url)
    os.environ["ACP_RUN_MODE"] = "join"

    result = run_tool_only_react(
        build_prompt(args.name, args.description),
        max_iterations=args.max_iterations,
        model=args.model,
    )
    print(f"\nCompleted after {result.iterations} iterations.")


if __name__ == "__main__":
    main()
