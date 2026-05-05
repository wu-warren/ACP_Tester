"""ReAct runner for ACP tool-only workflows."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

from agent.acp_tools import ACP_BACKEND_SKILL, ToolError, dispatch_tool
from agent.acp_tools import local_env_path
from agent.agent import Agent, create_client
from agent.tracing import parse_react_response


ACP_REACT_SYSTEM = """
You are an ACP game-playing ReAct agent.

You run in a loop of Thought, Action, PAUSE, Observation.
You must use tools for every external action. Do not claim that you read a file,
made an HTTP request, slept, or saved a key unless you called the matching tool.

Action format must be exactly one line:
Action: tool_name: argument

Available tools:
- read_skill: backend/SKILL.md
  Reads the only ACP skill file you may access.
- curl_request: {"method":"GET|POST","url":"absolute URL","headers":{},"json":{}}
  Makes a guarded curl request. The URL is blocked unless it is under the
  control-plane or data-plane URLs from Agent_ACP/cdk/.env.
- save_api_key_to_env: sk_agent_...
  Saves the ACP API key to this run's configured local env file as API_KEY.
- get_env_var: API_KEY
  Reads a whitelisted value from the process environment or this run's configured local env file.
- sleep_seconds: {"seconds":5}
  Sleeps before a polling retry.

Use compact one-line JSON for curl_request arguments.
When the workflow is complete, output Decision: followed by the concise result.

Target boundary policy:
- If the task prompt includes a session_id, only act on that exact competition
  and its matching data-plane game. Do not list, join, or play another
  competition.
- If the task prompt includes a tournament_id, only act on that exact tournament
  and child games discovered from GET /tournaments/<tournament_id>. Do not list
  active competitions or join unrelated competitions.
- If a join response says the agent already joined the provided target, treat it
  as joined and poll the same target. Never recover by switching to a different
  competition or tournament.

Tournament completion policy:
- For tournament_id tasks, do not output Decision when the tournament is merely
  waiting or in_progress. Keep polling the same tournament.
- If the tournament is in_progress but this agent has no active child game yet,
  sleep and poll GET /tournaments/<tournament_id> again.
- For tournament_id tasks, output Decision only after tournament.status is
  completed, or after a non-recoverable error.

Tester context policy:
- This tester compacts older transcript turns to reduce token usage.
- The full backend skill is injected below, so use it as the active gameplay rules.
""".strip()

MAX_CONTEXT_CHARS = 70000
RECENT_MESSAGE_COUNT = 8
MAX_RECENT_MESSAGE_CHARS = 1200
MAX_ROLLING_MESSAGES = 12
ROOT_DIR = Path(__file__).resolve().parents[1]
TRACE_DIR = ROOT_DIR / "traces"
JWT_PATTERN = re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")


@dataclass
class ReactRunResult:
    final_text: str
    iterations: int
    trace_path: str | None = None


def _shorten(text: str, limit: int = MAX_RECENT_MESSAGE_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[: int(limit * 0.65)]
    tail = text[-int(limit * 0.25):]
    return f"{head}\n...[compacted {len(text) - len(head) - len(tail)} chars]...\n{tail}"


def _redact_trace_value(value):
    if isinstance(value, str):
        return JWT_PATTERN.sub("<ACCESS_TOKEN_REDACTED>", value)
    if isinstance(value, list):
        return [_redact_trace_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_trace_value(item) for key, item in value.items()}
    return value


def _write_trace(trace: dict) -> str:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = TRACE_DIR / f"acp_react_trace_{timestamp}.json"
    path.write_text(
        json.dumps(_redact_trace_value(trace), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return str(path)


def _finish_trace(trace: dict, final_text: str, iterations: int, status: str) -> str:
    trace["completed_at"] = datetime.now(timezone.utc).isoformat()
    trace["final_text"] = final_text
    trace["iterations"] = iterations
    trace["status"] = status
    trace_path = _write_trace(trace)
    print(f"\nTrace written to {trace_path}")
    return trace_path


def _system_with_injected_skill() -> str:
    skill_text = ACP_BACKEND_SKILL.read_text(encoding="utf-8")
    return (
        f"{ACP_REACT_SYSTEM}\n\n"
        "Injected backend/SKILL.md:\n"
        "```markdown\n"
        f"{skill_text}\n"
        "```"
    )


def _summarize_observation_for_context(tool_name: str, observation: str) -> str:
    body_text, status = (observation.rsplit("\nHTTPSTATUS:", 1) + [""])[:2] if "\nHTTPSTATUS:" in observation else (observation, "")
    try:
        body = json.loads(body_text) if body_text.strip() else None
    except json.JSONDecodeError:
        body = None

    if isinstance(body, dict) and "tournament" in body:
        tournament = body.get("tournament") or {}
        viewer = body.get("viewer") or {}
        return json.dumps({
            "http_status": status.strip() or None,
            "tournament_status": tournament.get("status"),
            "counts": body.get("counts"),
            "viewer_instruction": viewer.get("instruction"),
            "active_child_session_ids": viewer.get("active_child_session_ids"),
            "should_join_tournament": viewer.get("should_join_tournament"),
            "should_wait_for_child_match": viewer.get("should_wait_for_child_match"),
        }, separators=(",", ":"))

    if isinstance(body, dict) and "access_token" in body:
        return json.dumps({"http_status": status.strip() or None, "access_token": "<ACCESS_TOKEN_SAVED>"}, separators=(",", ":"))

    if len(observation) > 1800:
        return _shorten(observation, 1800)
    return observation


def _context_size(agent: Agent, next_prompt: str) -> int:
    message_size = sum(len(str(message.get("content") or "")) for message in agent.messages)
    return len(agent.system) + message_size + len(next_prompt)


def _compact_and_require_skill(agent: Agent, task_prompt: str, next_prompt: str) -> str:
    recent_messages = agent.messages[-RECENT_MESSAGE_COUNT:]
    recent_summary = []
    for message in recent_messages:
        role = message.get("role", "unknown")
        content = _shorten(str(message.get("content") or ""))
        recent_summary.append(f"{role}: {content}")

    agent.messages = []
    return (
        "Observation: Context was compacted because it became too long. "
        "The full skill remains injected in the system context. Continue from the current task and observations.\n\n"
        f"Original task:\n{task_prompt}\n\n"
        f"Pending prompt before compaction:\n{_shorten(next_prompt)}\n\n"
        "Recent compacted transcript:\n"
        + "\n\n".join(recent_summary)
    )


def _compact_history(agent: Agent) -> None:
    if len(agent.messages) <= MAX_ROLLING_MESSAGES:
        return

    older_messages = agent.messages[:-MAX_ROLLING_MESSAGES]
    recent_messages = agent.messages[-MAX_ROLLING_MESSAGES:]
    compacted_turns = []
    for message in older_messages:
        role = message.get("role", "unknown")
        content = _shorten(str(message.get("content") or ""))
        compacted_turns.append(f"{role}: {content}")

    compacted_message = {
        "role": "user",
        "content": (
            "Observation: Earlier model context was compacted to limit token usage. "
            "Use this summary as continuity, and reread SKILL.md when exact rules matter.\n\n"
            "Compacted earlier transcript:\n"
            + "\n\n".join(compacted_turns)
        ),
    }
    agent.messages = [compacted_message] + recent_messages
    print(f"\n[tester notice] Compacted {len(older_messages)} older message(s) in model context to limit token usage.\n")


def run_tool_only_react(
    task_prompt: str,
    max_iterations: int = 80,
    model: str | None = None,
    skill_mode: str = "default",
) -> ReactRunResult:
    client = create_client()
    system = _system_with_injected_skill()
    agent = Agent(client=client, system=system, model=model)
    next_prompt = task_prompt
    trace = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "task_prompt": task_prompt,
        "model": model,
        "skill_mode": skill_mode,
        "max_iterations": max_iterations,
        "profile": os.getenv("ACP_TRACE_PROFILE") or None,
        "env_file": os.getenv("ACP_TRACE_ENV_FILE") or None,
        "local_env_path": str(local_env_path()),
        "target_session_id": os.getenv("ACP_TARGET_SESSION_ID") or None,
        "target_tournament_id": os.getenv("ACP_TARGET_TOURNAMENT_ID") or None,
        "system": system,
        "events": [],
    }

    iteration = 0
    try:
        for iteration in range(1, max_iterations + 1):
            if _context_size(agent, next_prompt) > MAX_CONTEXT_CHARS:
                next_prompt = _compact_and_require_skill(agent, task_prompt, next_prompt)
                trace["events"].append({
                    "iteration": iteration,
                    "type": "context_compaction_prompt",
                    "prompt": next_prompt,
                })

            trace["events"].append({
                "iteration": iteration,
                "type": "model_prompt",
                "content": next_prompt,
            })
            response = agent(next_prompt)
            print(response)
            trace["events"].append({
                "iteration": iteration,
                "type": "model_response",
                "content": response,
            })
            _compact_history(agent)
            parsed = parse_react_response(response)

            if parsed.get("decision") and not parsed.get("action"):
                trace_path = _finish_trace(trace, response, iteration, "completed")
                return ReactRunResult(final_text=response, iterations=iteration, trace_path=trace_path)

            if parsed.get("has_pause") and parsed.get("action"):
                tool_name = str(parsed["action"]).strip()
                argument = str(parsed.get("argument") or "").strip()
                trace["events"].append({
                    "iteration": iteration,
                    "type": "tool_call",
                    "tool_name": tool_name,
                    "argument": argument,
                })
                try:
                    observation = dispatch_tool(tool_name, argument)
                except ToolError as exc:
                    observation = f"Tool blocked/error: {exc}"
                print(f"\nObservation: {observation}\n")
                trace["events"].append({
                    "iteration": iteration,
                    "type": "tool_observation",
                    "tool_name": tool_name,
                    "observation": observation,
                })
                compact_observation = _summarize_observation_for_context(tool_name, observation)
                trace["events"].append({
                    "iteration": iteration,
                    "type": "model_context_observation",
                    "tool_name": tool_name,
                    "observation": compact_observation,
                })
                next_prompt = f"Observation: {compact_observation}"
                continue

            next_prompt = (
                "Observation: Invalid response. Use exactly Thought, then "
                "Action: tool_name: argument, then PAUSE; or Decision when complete."
            )
            trace["events"].append({
                "iteration": iteration,
                "type": "invalid_response_reprompt",
                "content": next_prompt,
            })
    except KeyboardInterrupt:
        final_text = "Interrupted by user before completion."
        trace["events"].append({
            "iteration": iteration,
            "type": "interrupted",
            "reason": "KeyboardInterrupt",
        })
        trace_path = _finish_trace(trace, final_text, iteration, "interrupted")
        raise KeyboardInterrupt(f"Interrupted by user. Trace JSON: {trace_path}")
    except Exception as exc:
        final_text = f"Stopped due to exception: {type(exc).__name__}: {exc}"
        trace["events"].append({
            "iteration": iteration,
            "type": "exception",
            "exception_type": type(exc).__name__,
            "message": str(exc),
        })
        _finish_trace(trace, final_text, iteration, "error")
        raise

    final_text = f"Stopped after max_iterations={max_iterations} without a final Decision."
    trace_path = _finish_trace(trace, final_text, max_iterations, "max_iterations")
    return ReactRunResult(final_text=final_text, iterations=max_iterations, trace_path=trace_path)
