#!/usr/bin/env python3
"""Run a tournament with deterministic orchestration and agent-selected moves.

This mirrors Agent_ACP/backend/src/tests/test-tournament-tictactoe.ts:
- admin creates a tournament
- four existing claimed agents join it
- this script polls active child matches
- each agent gets move tools only when its token has legal actions

Usage:
    python scripts/tournament_agent_moves.py
    python scripts/tournament_agent_moves.py --game-type connect_four
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


ROOT_DIR = Path(__file__).resolve().parents[1]
ARENA_DIR = ROOT_DIR.parent
ACP_CDK_ENV = ARENA_DIR / "Agent_ACP" / "cdk" / ".env"

try:
    from dotenv import dotenv_values, load_dotenv

    load_dotenv(ROOT_DIR / ".env")
    load_dotenv(ACP_CDK_ENV, override=False)
except ImportError:
    dotenv_values = None


_LOG_FILE: Any | None = None
_LOG_PATH: Path | None = None
_LOG_LOCK = threading.Lock()
_PRINT_LOCK = threading.Lock()


def init_jsonl_logger(path: Path | None) -> None:
    """Open the JSONL log file (truncating any prior run). No-op when path is None."""
    global _LOG_FILE, _LOG_PATH
    with _LOG_LOCK:
        if _LOG_FILE is not None:
            try:
                _LOG_FILE.close()
            except Exception:
                pass
            _LOG_FILE = None
            _LOG_PATH = None
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        _LOG_FILE = open(path, "w", encoding="utf-8")
        _LOG_PATH = path


def log_event(event: str, **fields: Any) -> None:
    """Append one JSON record to the log file. Silent if no logger is configured. Thread-safe."""
    if _LOG_FILE is None:
        return
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "thread": threading.current_thread().name,
    }
    for key, value in fields.items():
        record[key] = value
    line = json.dumps(record, default=str, separators=(",", ":")) + "\n"
    with _LOG_LOCK:
        if _LOG_FILE is None:
            return
        try:
            _LOG_FILE.write(line)
            _LOG_FILE.flush()
        except Exception:
            pass


def safe_print(*args: Any, **kwargs: Any) -> None:
    """Lock-protected print so parallel agents don't interleave mid-line."""
    with _PRINT_LOCK:
        print(*args, **kwargs)


BACKEND_URL = os.getenv(
    "BACKEND_SERVER_URL",
    "https://llw83cu38l.execute-api.us-west-2.amazonaws.com",
).rstrip("/")
GAMEAPI_URL = os.getenv(
    "GAMEAPI_SERVER_URL",
    "http://GameAp-Servi-tOMiXPVqPFIe-185462629.us-west-2.elb.amazonaws.com",
).rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TEST_ADMIN_EMAIL = os.getenv("TEST_ADMIN_EMAIL")
TEST_ADMIN_PASSWORD = os.getenv("TEST_ADMIN_PASSWORD")


@dataclass
class AgentClient:
    label: str
    api_key: str
    access_token: str
    auth_user_id: str
    user_id: str
    agent_id: str
    name: str


@dataclass
class AutonomousAgentSession:
    agent: AgentClient
    messages: list[dict[str, Any]]
    finished: bool = False
    steps: int = 0


def env_file_value(path: Path, key: str) -> str:
    if dotenv_values is None or not path.exists():
        return ""
    return dotenv_values(path).get(key, "") or ""


def require_value(name: str, value: str | None) -> str:
    if not value:
        raise SystemExit(f"Missing required value: {name}")
    return value


def raise_for_status_with_body(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text.strip()
        if body:
            raise requests.HTTPError(f"{exc}; response body: {body}", response=response) from exc
        raise


def request_json(
    method: str,
    url: str,
    *,
    token: str | None = None,
    payload: dict[str, Any] | None = None,
    timeout: int = 20,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.request(method, url, json=payload, headers=headers, timeout=timeout)
    raise_for_status_with_body(response)
    return response.json() if response.text.strip() else {}


def decode_jwt_sub(token: str) -> str:
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    decoded = base64.urlsafe_b64decode(payload)
    data = json.loads(decoded)
    return data["sub"]


def agent_key(suffix: str) -> str:
    env_name = f"AGENT_{suffix.upper()}_API_KEY"
    return (
        os.getenv(env_name)
        or os.getenv(f"TEST_AGENT_{suffix.upper()}_API_KEY")
        or env_file_value(ROOT_DIR / f".env.agent-{suffix.lower()}", "API_KEY")
    )


def login_admin() -> str:
    email = require_value("TEST_ADMIN_EMAIL", TEST_ADMIN_EMAIL)
    password = require_value("TEST_ADMIN_PASSWORD", TEST_ADMIN_PASSWORD)
    data = request_json(
        "POST",
        f"{BACKEND_URL}/auth/human/login",
        payload={"email": email, "password": password},
    )
    return data["access_token"]


def login_agent(label: str, api_key: str) -> AgentClient:
    login = request_json(
        "POST",
        f"{BACKEND_URL}/auth/agent/login",
        payload={"api_key": api_key},
    )
    token = login["access_token"]
    me = request_json("GET", f"{BACKEND_URL}/auth/agent/me", token=token)
    status = me.get("status")
    if status != "claimed":
        raise RuntimeError(f"{label} is not claimed (status={status or 'unknown'})")
    return AgentClient(
        label=label,
        api_key=api_key,
        access_token=token,
        auth_user_id=decode_jwt_sub(token),
        user_id=me.get("user_id", ""),
        agent_id=me["id"],
        name=me["name"],
    )


def create_tournament(
    admin_token: str,
    game_type: str,
    max_participants: int,
    max_active_matches: int,
) -> str:
    data = request_json(
        "POST",
        f"{BACKEND_URL}/admin/tournaments/create",
        token=admin_token,
        payload={
            "game_type": game_type,
            "max_participants": max_participants,
            "max_active_matches": max_active_matches,
            "metadata": {"runner": "tournament_agent_moves"},
        },
    )
    return data["tournament_id"]


def join_tournament(tournament_id: str, agent: AgentClient) -> None:
    try:
        request_json(
            "POST",
            f"{BACKEND_URL}/tournaments/{tournament_id}/join",
            token=agent.access_token,
            payload={},
        )
    except requests.HTTPError as exc:
        if "already joined" in str(exc).lower():
            return
        raise


def compact_tournament(details: dict[str, Any]) -> dict[str, Any]:
    tournament = details.get("tournament") or {}
    compact: dict[str, Any] = {
        "tournament": {
            "tournament_id": tournament.get("tournament_id"),
            "status": tournament.get("status"),
            "game_type": tournament.get("game_type"),
            "game_server_url": tournament.get("game_server_url"),
        },
        "counts": details.get("counts"),
    }

    viewer = details.get("viewer")
    if isinstance(viewer, dict):
        compact["viewer"] = {
            "agent_id": viewer.get("agent_id"),
            "is_tournament_participant": viewer.get("is_tournament_participant"),
            "active_child_session_ids": viewer.get("active_child_session_ids") or [],
            "should_join_tournament": viewer.get("should_join_tournament"),
            "should_wait_for_child_match": viewer.get("should_wait_for_child_match"),
            "instruction": viewer.get("instruction"),
        }
        compact["active_child_matches"] = [
            {
                "session_id": match.get("session_id"),
                "status": match.get("status"),
                "game_type": match.get("game_type"),
            }
            for match in details.get("active_child_matches", [])
        ]
    else:
        compact["competitions"] = [
            {
                "session_id": comp.get("session_id"),
                "status": comp.get("status"),
                "game_type": comp.get("game_type"),
            }
            for comp in details.get("competitions", [])
        ]

    return compact


def agent_join_tournament_with_tools(
    tournament_id: str,
    agent: AgentClient,
    verbose_llm: bool,
) -> None:
    """Let the LLM use control-plane tools to join a tournament."""
    if not OPENAI_API_KEY:
        join_tournament(tournament_id, agent)
        return

    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_identity",
                "description": "Return this agent's claimed identity.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_tournament",
                "description": (
                    "Get the tournament view scoped to this agent. The response includes "
                    "viewer.is_tournament_participant and viewer.should_join_tournament so you "
                    "can tell whether you still need to call join_tournament."
                ),
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "join_tournament",
                "description": "Join this agent to the target tournament.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
    ]
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are an ACP tournament agent. You can only join or inspect the tournament by calling tools. "
                "Your immediate task is to join the target tournament. Call join_tournament exactly once unless "
                "a tool result says you are already joined."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Agent: {agent.name}\n"
                f"Agent label: {agent.label}\n"
                f"Tournament: {tournament_id}\n"
                "Use the available tools to join this tournament."
            ),
        },
    ]
    print_llm_event(verbose_llm, f"{agent.label} join system prompt", messages[0]["content"])
    print_llm_event(verbose_llm, f"{agent.label} join user prompt", messages[1]["content"])

    for iteration in range(1, 7):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        message = response.choices[0].message
        message_dump = message.model_dump(exclude_none=True)
        messages.append(message_dump)
        print_llm_event(verbose_llm, f"{agent.label} join llm output #{iteration}", message_dump)

        tool_calls = message.tool_calls or []
        if not tool_calls:
            print(f"   {agent.label} produced no join tool call; fallback joins directly")
            join_tournament(tournament_id, agent)
            return

        for tool_call in tool_calls:
            name = tool_call.function.name
            args = parse_tool_arguments(tool_call.function.arguments)
            print_llm_event(
                verbose_llm,
                f"{agent.label} join tool call #{iteration}",
                {"name": name, "arguments": args},
            )

            if name == "get_identity":
                tool_result = {
                    "agent_id": agent.agent_id,
                    "user_id": agent.user_id,
                    "name": agent.name,
                    "status": "claimed",
                }
            elif name == "get_tournament":
                tool_result = compact_tournament(get_tournament(tournament_id, token=agent.access_token))
            elif name == "join_tournament":
                try:
                    join_tournament(tournament_id, agent)
                    tool_result = {"joined": True, "tournament_id": tournament_id}
                except requests.HTTPError as exc:
                    if "already joined" in str(exc).lower():
                        tool_result = {"joined": True, "already_joined": True, "tournament_id": tournament_id}
                    else:
                        raise
                print(f"   {agent.label} ({agent.name}) tool join_tournament()")
                print_llm_event(verbose_llm, f"{agent.label} join tool result #{iteration}", tool_result)
                return
            else:
                tool_result = {"error": f"unknown tool {name}"}

            print_llm_event(verbose_llm, f"{agent.label} join tool result #{iteration}", tool_result)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result, separators=(",", ":")),
            })

    print(f"   {agent.label} join tool loop timed out; fallback joins directly")
    join_tournament(tournament_id, agent)


def get_tournament(tournament_id: str, token: str | None = None) -> dict[str, Any]:
    return request_json("GET", f"{BACKEND_URL}/tournaments/{tournament_id}", token=token)


def autonomous_agent_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_identity",
                "description": "Return this agent's claimed identity.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_tournament",
                "description": (
                    "Get the tournament view scoped to this agent. The response includes "
                    "viewer.active_child_session_ids (the session_ids this agent must play), "
                    "viewer.is_tournament_participant, viewer.should_join_tournament, "
                    "viewer.should_wait_for_child_match, viewer.instruction (next-step guidance), "
                    "and active_child_matches (this agent's currently-playable matches). Never call "
                    "get_game_state or submit_move on any session_id outside viewer.active_child_session_ids."
                ),
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "join_tournament",
                "description": "Join this agent to the target tournament.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_game_state",
                "description": "Get this agent's game state for a tournament child match.",
                "parameters": {
                    "type": "object",
                    "properties": {"session_id": {"type": "string"}},
                    "required": ["session_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_move",
                "description": "Submit one legal action for this agent in a tournament child match.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "action": {"type": "integer"},
                    },
                    "required": ["session_id", "action"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "sleep",
                "description": (
                    "Pause this agent for a short interval before its next tool call. Choose the "
                    "duration based on what you are waiting for:\n"
                    "- 1-2 seconds: you are in the opponent's turn of YOUR active match (legal_actions "
                    "came back empty, is_terminal is false). Opponents move quickly; you want to be "
                    "responsive when it becomes your turn.\n"
                    "- ~30 seconds: viewer.active_child_session_ids is empty OR "
                    "viewer.should_wait_for_child_match is true. You are waiting for the orchestrator "
                    "to assign you a match, which is much slower.\n"
                    "The seconds argument is optional and defaults to 30; values are clamped to [1, 60]."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "seconds": {
                            "type": "number",
                            "description": (
                                "How long to sleep, in seconds. Use ~1-2 when polling the opponent's "
                                "turn in your own match. Use ~30 when waiting for a match to be "
                                "assigned. Defaults to 30. Clamped to [1, 60]."
                            ),
                        },
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Declare that this agent is done because the tournament is completed.",
                "parameters": {
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def make_autonomous_session(
    agent: AgentClient,
    tournament_id: str,
    game_type: str,
) -> AutonomousAgentSession:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an ACP tournament-playing agent with tools. You receive exactly one target "
                "tournament_id and must not switch to any other tournament. Use tools for all external "
                "actions.\n\n"
                "Workflow:\n"
                "1. Call get_tournament. The response is scoped to you: it contains a `viewer` object with "
                "`is_tournament_participant`, `should_join_tournament`, `should_wait_for_child_match`, "
                "`active_child_session_ids`, and a human-readable `instruction`. It also contains "
                "`active_child_matches`, the list of child matches you are currently playing.\n"
                "2. If `viewer.should_join_tournament` is true, call join_tournament, then call "
                "get_tournament again.\n"
                "3. For each session_id in `viewer.active_child_session_ids`, call get_game_state on "
                "that session_id. Then:\n"
                "   - If `legal_actions` is non-empty, choose one legal action and call submit_move on "
                "that same session_id.\n"
                "   - If `legal_actions` is empty AND `is_terminal` is false, it is the OPPONENT'S TURN "
                "in YOUR match. You are still a player in that match. Do NOT abandon the session_id. "
                "Call sleep with `seconds: 1` (or at most 2), then call get_game_state on the SAME "
                "session_id again to see if the opponent has moved. Use the SHORT sleep here so you "
                "stay responsive — opponents typically move within a couple of seconds. Repeat until "
                "`legal_actions` becomes non-empty (your turn) or `is_terminal` is true (the match "
                "ended).\n"
                "   - If `is_terminal` is true, the match is over. Move on to the next session_id in "
                "`viewer.active_child_session_ids`, or refresh by calling get_tournament if there are "
                "no more.\n"
                "4. Empty `legal_actions` NEVER means \"this is not your session\". A session_id that "
                "appears in `viewer.active_child_session_ids` belongs to you for the ENTIRE match, on "
                "every turn including the opponent's. The backend keeps it in that list until the match "
                "is terminal.\n"
                "5. NEVER call get_game_state or submit_move on a session_id that is NOT in "
                "`viewer.active_child_session_ids`. Those sessions belong to other agents and will "
                "return 401/403/404.\n"
                "6. If `viewer.active_child_session_ids` is empty and the tournament is not completed, "
                "call sleep with `seconds: 30` (the default — waiting for a match takes much longer "
                "than an opponent move) and THEN call get_tournament again. Do not poll in a tight "
                "loop.\n"
                "7. When `tournament.status` is `completed`, call finish.\n\n"
                "Do not invent session IDs, moves, or results."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Agent: {agent.name}\n"
                f"Agent label: {agent.label}\n"
                f"Game type: {game_type}\n"
                f"Target tournament_id: {tournament_id}\n"
                "Join and play this tournament using only your tools."
            ),
        },
    ]
    return AutonomousAgentSession(agent=agent, messages=messages)


def report_competition_complete(
    session_id: str,
    terminal_state: dict[str, Any],
    agents_by_name: dict[str, AgentClient],
) -> None:
    returns = terminal_state.get("returns") or {}
    results: dict[str, float] = {}
    for player_name, score in returns.items():
        agent = agents_by_name.get(player_name)
        if not agent:
            raise RuntimeError(f"No local agent matched terminal player {player_name!r}")
        results[agent.auth_user_id] = float(score)

    request_json(
        "POST",
        f"{BACKEND_URL}/internal/competitions/{session_id}/complete",
        payload={
            "results": results,
            "termination_reason": terminal_state.get("termination_reason"),
        },
    )


def extract_legal_actions(state: dict[str, Any]) -> list[int]:
    actions = state.get("legal_actions")
    if not isinstance(actions, list):
        return []
    return [int(action) for action in actions if isinstance(action, int) or str(action).isdigit()]


def heuristic_move(game_type: str, legal_actions: list[int]) -> int:
    if not legal_actions:
        raise RuntimeError("Cannot choose a move without legal actions")
    priorities = {
        "tic_tac_toe": [4, 0, 2, 6, 8, 1, 3, 5, 7],
        "connect_four": [3, 4, 2, 5, 1, 6, 0],
    }
    for action in priorities.get(game_type, []):
        if action in legal_actions:
            return action
    return legal_actions[0]


def prompt_move(game_type: str, state: dict[str, Any], agent: AgentClient) -> int:
    legal_actions = extract_legal_actions(state)
    if not OPENAI_API_KEY:
        return heuristic_move(game_type, legal_actions)

    from openai import OpenAI

    prompt = f"""You are {agent.name} playing {game_type}.

Observation:
{state.get("observation", "")}

Legal actions: {legal_actions}

Choose the best move. Reply with only one legal action number."""

    response = OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
    )
    move_text = response.choices[0].message.content or ""
    for match in re.findall(r"\d+", move_text):
        action = int(match)
        if action in legal_actions:
            return action
    return heuristic_move(game_type, legal_actions)


def compact_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": state.get("session_id"),
        "game_name": state.get("game_name"),
        "status": state.get("status"),
        "observation": state.get("observation"),
        "current_player": state.get("current_player"),
        "legal_actions": state.get("legal_actions"),
        "legal_actions_str": state.get("legal_actions_str"),
        "is_terminal": state.get("is_terminal"),
        "returns": state.get("returns"),
        "move_count": state.get("move_count"),
        "termination_reason": state.get("termination_reason"),
    }


def print_llm_event(enabled: bool, title: str, value: Any) -> None:
    if not enabled:
        return
    body = value if isinstance(value, str) else json.dumps(value, indent=2)
    block = f"\n--- {title} ---\n{body}\n--- end ---\n"
    with _PRINT_LOCK:
        print(block)


def parse_tool_arguments(raw_arguments: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def tool_agent_turn(
    session_id: str,
    game_type: str,
    agent: AgentClient,
    initial_state: dict[str, Any],
    verbose_llm: bool,
) -> dict[str, Any]:
    """Let the LLM use turn-scoped tools to submit a move."""
    if not OPENAI_API_KEY:
        action = heuristic_move(game_type, extract_legal_actions(initial_state))
        return submit_move(session_id, agent.access_token, action)

    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_game_state",
                "description": "Get this agent's current game state for the active child match.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_move",
                "description": "Submit one legal action for this agent in the active child match.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "integer",
                            "description": "One integer from the latest legal_actions list.",
                        }
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
            },
        },
    ]
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a game-playing agent. You can only affect the game by calling tools. "
                "Before moving, inspect legal_actions. Choose exactly one legal action and call submit_move. "
                "Do not describe a move without calling submit_move."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Agent: {agent.name}\n"
                f"Game: {game_type}\n"
                f"Session: {session_id}\n"
                "Initial state from the orchestrator:\n"
                f"{json.dumps(compact_state(initial_state), separators=(',', ':'))}"
            ),
        },
    ]
    print_llm_event(verbose_llm, f"{agent.label} system prompt", messages[0]["content"])
    print_llm_event(verbose_llm, f"{agent.label} user prompt", messages[1]["content"])

    for iteration in range(1, 7):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        message = response.choices[0].message
        message_dump = message.model_dump(exclude_none=True)
        messages.append(message_dump)
        print_llm_event(verbose_llm, f"{agent.label} llm output #{iteration}", message_dump)
        tool_calls = message.tool_calls or []

        if not tool_calls:
            action = prompt_move(game_type, initial_state, agent)
            print(f"   {agent.label} produced no tool call; fallback submits {action}")
            return submit_move(session_id, agent.access_token, action)

        for tool_call in tool_calls:
            name = tool_call.function.name
            args = parse_tool_arguments(tool_call.function.arguments)
            print_llm_event(
                verbose_llm,
                f"{agent.label} tool call #{iteration}",
                {"name": name, "arguments": args},
            )

            if name == "get_game_state":
                tool_result = compact_state(get_game_state(session_id, agent.access_token))
            elif name == "submit_move":
                action = int(args["action"])
                tool_result = compact_state(submit_move(session_id, agent.access_token, action))
                print(f"   {agent.label} ({agent.name}) tool submit_move({action})")
                print_llm_event(verbose_llm, f"{agent.label} tool result #{iteration}", tool_result)
                return tool_result
            else:
                tool_result = {"error": f"unknown tool {name}"}

            print_llm_event(verbose_llm, f"{agent.label} tool result #{iteration}", tool_result)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result, separators=(",", ":")),
            })

    action = prompt_move(game_type, initial_state, agent)
    print(f"   {agent.label} tool loop timed out; fallback submits {action}")
    return submit_move(session_id, agent.access_token, action)


def get_game_state(session_id: str, token: str) -> dict[str, Any]:
    return request_json("GET", f"{GAMEAPI_URL}/games/{session_id}", token=token)


def submit_move(session_id: str, token: str, action: int) -> dict[str, Any]:
    return request_json(
        "POST",
        f"{GAMEAPI_URL}/games/{session_id}/step",
        token=token,
        payload={"action": action},
    )


def execute_autonomous_tool(
    session: AutonomousAgentSession,
    tournament_id: str,
    agents_by_name: dict[str, AgentClient],
    tool_name: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    agent = session.agent

    if tool_name == "get_identity":
        return {
            "agent_id": agent.agent_id,
            "user_id": agent.user_id,
            "name": agent.name,
            "status": "claimed",
        }

    if tool_name == "get_tournament":
        return compact_tournament(get_tournament(tournament_id, token=agent.access_token))

    if tool_name == "join_tournament":
        try:
            join_tournament(tournament_id, agent)
            return {"joined": True, "tournament_id": tournament_id}
        except requests.HTTPError as exc:
            if "already joined" in str(exc).lower():
                return {"joined": True, "already_joined": True, "tournament_id": tournament_id}
            raise

    if tool_name == "get_game_state":
        session_id = str(args.get("session_id") or "")
        if not session_id:
            return {"error": "session_id is required"}
        try:
            state = get_game_state(session_id, agent.access_token)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            return {"error": str(exc), "status_code": status}
        result = compact_state(state)
        if state.get("is_terminal"):
            report_competition_complete(session_id, state, agents_by_name)
            result["reported_competition_complete"] = True
        return result

    if tool_name == "submit_move":
        session_id = str(args.get("session_id") or "")
        if not session_id:
            return {"error": "session_id is required"}
        try:
            action = int(args["action"])
        except (KeyError, TypeError, ValueError):
            return {"error": "integer action is required"}
        state = submit_move(session_id, agent.access_token, action)
        result = compact_state(state)
        if state.get("is_terminal"):
            report_competition_complete(session_id, state, agents_by_name)
            result["reported_competition_complete"] = True
        return result

    if tool_name == "sleep":
        try:
            requested = float(args.get("seconds") if args.get("seconds") is not None else 30)
        except (TypeError, ValueError):
            requested = 30.0
        seconds = max(1.0, min(60.0, requested))
        log_event(
            "agent_sleep",
            agent_label=agent.label,
            agent_name=agent.name,
            requested_seconds=requested,
            actual_seconds=seconds,
        )
        print(f"   {agent.label} sleeping {seconds:.1f}s")
        time.sleep(seconds)
        return {"slept_seconds": seconds, "requested_seconds": requested}

    if tool_name == "finish":
        session.finished = True
        return {"finished": True, "summary": str(args.get("summary") or "")}

    return {"error": f"unknown tool {tool_name}"}


def autonomous_agent_step(
    session: AutonomousAgentSession,
    tournament_id: str,
    agents_by_name: dict[str, AgentClient],
    verbose_llm: bool,
) -> None:
    if session.finished:
        return
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for --agent-mode autonomous")

    from openai import OpenAI

    session.steps += 1
    agent = session.agent
    log_event(
        "agent_step_start",
        agent_label=agent.label,
        agent_name=agent.name,
        step=session.steps,
    )
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=session.messages,
            tools=autonomous_agent_tools(),
            tool_choice="auto",
        )
    except Exception as exc:
        log_event(
            "agent_llm_error",
            agent_label=agent.label,
            agent_name=agent.name,
            step=session.steps,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        print(f"   {agent.label} LLM call failed: {type(exc).__name__}: {exc}")
        session.messages.append({
            "role": "user",
            "content": f"Observation: previous LLM call failed ({type(exc).__name__}: {exc}). Try again next round.",
        })
        return

    message = response.choices[0].message
    message_dump = message.model_dump(exclude_none=True)
    session.messages.append(message_dump)
    print_llm_event(verbose_llm, f"{agent.label} autonomous llm output #{session.steps}", message_dump)
    log_event(
        "agent_llm_output",
        agent_label=agent.label,
        agent_name=agent.name,
        step=session.steps,
        content=(message_dump.get("content") or "")[:1000],
        tool_call_names=[tc.function.name for tc in (message.tool_calls or [])],
    )

    tool_calls = message.tool_calls or []
    if not tool_calls:
        observation = {
            "error": "No tool call was made. Continue by calling a tool.",
            "expected_tools": [
                "get_identity",
                "join_tournament",
                "get_tournament",
                "get_game_state",
                "submit_move",
                "sleep",
                "finish",
            ],
        }
        session.messages.append({"role": "user", "content": f"Observation: {json.dumps(observation)}"})
        log_event(
            "agent_no_tool_call",
            agent_label=agent.label,
            agent_name=agent.name,
            step=session.steps,
        )
        return

    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        args = parse_tool_arguments(tool_call.function.arguments)
        print_llm_event(
            verbose_llm,
            f"{agent.label} autonomous tool call #{session.steps}",
            {"name": tool_name, "arguments": args},
        )
        log_event(
            "agent_tool_call",
            agent_label=agent.label,
            agent_name=agent.name,
            step=session.steps,
            tool=tool_name,
            arguments=args,
        )
        try:
            tool_result = execute_autonomous_tool(session, tournament_id, agents_by_name, tool_name, args)
        except Exception as exc:
            tool_result = {
                "error": f"tool {tool_name} raised {type(exc).__name__}: {exc}",
                "error_type": type(exc).__name__,
            }
            log_event(
                "agent_tool_exception",
                agent_label=agent.label,
                agent_name=agent.name,
                step=session.steps,
                tool=tool_name,
                arguments=args,
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            print(f"   {agent.label} tool {tool_name} raised: {type(exc).__name__}: {exc}")
        print(f"   {agent.label} tool {tool_name}({json.dumps(args, separators=(',', ':'))})")
        print_llm_event(
            verbose_llm,
            f"{agent.label} autonomous tool result #{session.steps}",
            tool_result,
        )
        log_event(
            "agent_tool_result",
            agent_label=agent.label,
            agent_name=agent.name,
            step=session.steps,
            tool=tool_name,
            result=tool_result,
        )
        session.messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(tool_result, separators=(",", ":")),
        })


def play_match_to_terminal(
    session_id: str,
    game_type: str,
    agents: list[AgentClient],
    agents_by_name: dict[str, AgentClient],
    poll_seconds: float,
    move_mode: str,
    verbose_llm: bool,
) -> None:
    print(f"Playing child match: {session_id}")
    while True:
        acted = False
        for agent in agents:
            try:
                state = get_game_state(session_id, agent.access_token)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status in {401, 403, 404}:
                    continue
                raise

            if state.get("is_terminal"):
                report_competition_complete(session_id, state, agents_by_name)
                return

            legal_actions = extract_legal_actions(state)
            if not legal_actions:
                continue

            if move_mode == "tool":
                print(f"   {agent.label} ({agent.name}) has legal actions {legal_actions}; starting tool-using LLM turn")
                step_state = tool_agent_turn(session_id, game_type, agent, state, verbose_llm)
            else:
                action = (
                    prompt_move(game_type, state, agent)
                    if move_mode == "prompt"
                    else heuristic_move(game_type, legal_actions)
                )
                print(f"   {agent.label} ({agent.name}) plays {action}; legal={legal_actions}")
                step_state = submit_move(session_id, agent.access_token, action)
            acted = True
            if step_state.get("is_terminal"):
                report_competition_complete(session_id, step_state, agents_by_name)
                return

        if not acted:
            time.sleep(poll_seconds)


def wait_for_active_child(tournament_id: str, poll_seconds: float, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        details = get_tournament(tournament_id)
        counts = details.get("counts") or {}
        if counts.get("active", 0) > 0:
            return
        time.sleep(poll_seconds)
    raise TimeoutError(f"Timed out waiting for tournament {tournament_id} to start a child match")


def print_tournament_complete(details: dict[str, Any]) -> None:
    tournament = details.get("tournament") or {}
    counts = details.get("counts") or {}
    print("\nTournament completed")
    print(f"Completed matches: {counts.get('completed')}")
    print(f"Queued: {counts.get('queued')} Active: {counts.get('active')}")
    print("Leaderboard:")
    print(json.dumps(tournament.get("leaderboard", []), indent=2))


def run_autonomous_tournament(
    args: argparse.Namespace,
    tournament_id: str,
    agents: list[AgentClient],
) -> None:
    agents_by_name = {agent.name: agent for agent in agents}
    sessions = [
        make_autonomous_session(agent, tournament_id, args.game_type)
        for agent in agents
    ]
    for session in sessions:
        print_llm_event(args.verbose_llm, f"{session.agent.label} autonomous system prompt", session.messages[0]["content"])
        print_llm_event(args.verbose_llm, f"{session.agent.label} autonomous user prompt", session.messages[1]["content"])

    def _run_session_step(session: AutonomousAgentSession, round_number: int) -> None:
        if session.finished:
            return
        try:
            autonomous_agent_step(session, tournament_id, agents_by_name, args.verbose_llm)
        except Exception as exc:
            log_event(
                "agent_step_exception",
                agent_label=session.agent.label,
                agent_name=session.agent.name,
                step=session.steps,
                round=round_number,
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            safe_print(
                f"   {session.agent.label} step raised {type(exc).__name__}: {exc}; "
                "continuing with other agents"
            )

    with ThreadPoolExecutor(
        max_workers=max(1, len(sessions)),
        thread_name_prefix="agent",
    ) as executor:
        for step in range(1, args.max_agent_steps + 1):
            safe_print(f"\n=== Autonomous agent round {step} ===")
            log_event("round_start", round=step)

            active_sessions = [s for s in sessions if not s.finished]
            if active_sessions:
                futures = [
                    executor.submit(_run_session_step, session, step)
                    for session in active_sessions
                ]
                for future in futures:
                    future.result()

            details = get_tournament(tournament_id)
            tournament = details.get("tournament") or {}
            counts = details.get("counts") or {}
            log_event(
                "round_end",
                round=step,
                tournament_status=tournament.get("status"),
                counts=counts,
                finished_agents=[s.agent.label for s in sessions if s.finished],
            )
            if tournament.get("status") == "completed":
                print_tournament_complete(details)
                log_event("tournament_completed", tournament_id=tournament_id)
                return

            time.sleep(args.poll_seconds)

    log_event(
        "max_agent_steps_reached",
        tournament_id=tournament_id,
        max_agent_steps=args.max_agent_steps,
    )
    raise TimeoutError(
        f"Stopped after --max-agent-steps={args.max_agent_steps}; "
        f"tournament {tournament_id} was not completed."
    )


def run(args: argparse.Namespace) -> None:
    log_path: Path | None = None
    if getattr(args, "log_file", None):
        log_path = Path(args.log_file).expanduser().resolve()
    init_jsonl_logger(log_path)

    print("Tournament Agent Moves")
    print(f"Backend : {BACKEND_URL}")
    print(f"GameAPI : {GAMEAPI_URL}")
    print(f"Game    : {args.game_type}")
    print(f"Join    : {args.join_mode}")
    print(f"Moves   : {args.move_mode}")
    print(f"Agent   : {args.agent_mode}")
    if log_path is not None:
        print(f"Log     : {log_path}")
    print()

    log_event(
        "run_start",
        backend=BACKEND_URL,
        gameapi=GAMEAPI_URL,
        game_type=args.game_type,
        agent_mode=args.agent_mode,
        join_mode=args.join_mode,
        move_mode=args.move_mode,
        max_active_matches=args.max_active_matches,
        max_agent_steps=args.max_agent_steps,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
        tournament_id=args.tournament_id,
    )

    admin_token = login_admin()
    print("Admin logged in")

    agents = [
        login_agent(f"Agent-{suffix.upper()}", require_value(f".env.agent-{suffix} API_KEY", agent_key(suffix)))
        for suffix in ("a", "b", "c", "d")
    ]
    for agent in agents:
        print(f"   {agent.label}: claimed as {agent.name}")

    tournament_id = args.tournament_id or create_tournament(
        admin_token,
        args.game_type,
        max_participants=len(agents),
        max_active_matches=args.max_active_matches,
    )
    print(f"\nTournament: {tournament_id}")

    if args.agent_mode == "autonomous":
        run_autonomous_tournament(args, tournament_id, agents)
        return

    for agent in agents:
        if args.join_mode == "tool":
            agent_join_tournament_with_tools(tournament_id, agent, args.verbose_llm)
        else:
            join_tournament(tournament_id, agent)
        print(f"   {agent.label} joined")

    wait_for_active_child(tournament_id, args.poll_seconds, args.timeout_seconds)

    agents_by_name = {agent.name: agent for agent in agents}
    finished_sessions: set[str] = set()

    while True:
        details = get_tournament(tournament_id)
        tournament = details.get("tournament") or {}
        competitions = details.get("competitions") or []
        active = [c for c in competitions if c.get("status") == "in_progress"]

        for match in active:
            session_id = match["session_id"]
            if session_id in finished_sessions:
                continue
            play_match_to_terminal(
                session_id,
                args.game_type,
                agents,
                agents_by_name,
                args.poll_seconds,
                args.move_mode,
                args.verbose_llm,
            )
            finished_sessions.add(session_id)

        if tournament.get("status") == "completed":
            print_tournament_complete(details)
            return

        time.sleep(args.poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministically run a tournament while agents choose moves."
    )
    parser.add_argument("--tournament-id", help="Use an existing tournament instead of creating one.")
    parser.add_argument("--game-type", default="tic_tac_toe")
    parser.add_argument(
        "--agent-mode",
        choices=["turn", "autonomous"],
        default="turn",
        help="turn uses scoped join/move prompts; autonomous gives each agent one persistent prompt.",
    )
    parser.add_argument("--join-mode", choices=["tool", "direct"], default="tool")
    parser.add_argument("--move-mode", choices=["tool", "prompt", "heuristic"], default="tool")
    parser.add_argument("--quiet-llm", dest="verbose_llm", action="store_false", help="Hide LLM prompts, outputs, tool calls, and tool results.")
    parser.add_argument("--max-active-matches", type=int, default=1)
    parser.add_argument("--max-agent-steps", type=int, default=200)
    parser.add_argument("--poll-seconds", type=float, default=1.5)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument(
        "--log-file",
        default=str(ROOT_DIR / "logs" / "tournament_agent_moves.jsonl"),
        help="Path to a JSONL log file written this run (overwritten each run). Pass '' to disable.",
    )
    parser.set_defaults(verbose_llm=True)
    args = parser.parse_args()
    if args.log_file == "":
        args.log_file = None

    try:
        run(args)
    except Exception as exc:
        log_event(
            "run_failed",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        print(f"\nTournament run failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
