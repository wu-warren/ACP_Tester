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
from agent.acp_tools import allowed_base_urls


def build_prompt(session_id: str) -> str:
    control_plane_url = allowed_base_urls()["control_plane"].rstrip("/")
    return (
        f"read skill.md and play altruAgent session_id {session_id}. "
        f"Use this exact control-plane base URL for backend calls: {control_plane_url}. "
        f"Call {control_plane_url}/auth/agent/login, {control_plane_url}/auth/agent/me, "
        f"{control_plane_url}/competitions/{session_id}/join, and "
        f"{control_plane_url}/competitions/{session_id}. "
        "The session_id is already provided; do not list active competitions or fetch competition history."
    )


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
    parser = argparse.ArgumentParser(description="Join and play an ACP game session.")
    parser.add_argument("session_id")
    parser.add_argument("--profile", help="Use .env.<profile> for this agent's ACP API_KEY.")
    parser.add_argument("--env-file", help="Use this env file for this agent's ACP API_KEY.")
    parser.add_argument("--local-backend", action="store_true", help="Use http://localhost:3000 as the control plane.")
    parser.add_argument("--backend-url", help="Use this control-plane base URL instead of Agent_ACP/cdk/.env.")
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    configure_profile(args.profile, args.env_file)
    configure_backend(args.local_backend, args.backend_url)
    os.environ["ACP_RUN_MODE"] = "play"
    os.environ["ACP_TARGET_SESSION_ID"] = args.session_id
    if args.profile:
        os.environ["ACP_TRACE_PROFILE"] = args.profile
    if args.env_file:
        os.environ["ACP_TRACE_ENV_FILE"] = args.env_file

    result = run_tool_only_react(
        build_prompt(args.session_id),
        max_iterations=args.max_iterations,
        model=args.model,
        skill_mode="game",
    )
    print(f"\nCompleted after {result.iterations} iterations.")
    if result.trace_path:
        print(f"Trace JSON: {result.trace_path}")


if __name__ == "__main__":
    main()
