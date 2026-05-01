"""ReAct runner for ACP tool-only workflows."""

from __future__ import annotations

from dataclasses import dataclass

from agent.acp_tools import ToolError, dispatch_tool
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

Tester context policy:
- This tester keeps only a rolling recent transcript to reduce token usage.
- If older messages are not visible, continue from the current Observation and reread SKILL.md when needed.
""".strip()

MAX_CONTEXT_CHARS = 70000
RECENT_MESSAGE_COUNT = 8
MAX_RECENT_MESSAGE_CHARS = 1200
MAX_ROLLING_MESSAGES = 12


@dataclass
class ReactRunResult:
    final_text: str
    iterations: int


def _shorten(text: str, limit: int = MAX_RECENT_MESSAGE_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[: int(limit * 0.65)]
    tail = text[-int(limit * 0.25):]
    return f"{head}\n...[compacted {len(text) - len(head) - len(tail)} chars]...\n{tail}"


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
        "Before continuing, read the full skill again with `Action: read_skill: backend/SKILL.md`.\n\n"
        f"Original task:\n{task_prompt}\n\n"
        f"Pending prompt before compaction:\n{_shorten(next_prompt)}\n\n"
        "Recent compacted transcript:\n"
        + "\n\n".join(recent_summary)
    )


def _trim_history(agent: Agent) -> None:
    if len(agent.messages) <= MAX_ROLLING_MESSAGES:
        return
    dropped = len(agent.messages) - MAX_ROLLING_MESSAGES
    agent.messages = agent.messages[-MAX_ROLLING_MESSAGES:]
    print(f"\n[tester notice] Dropped {dropped} older message(s) from model context to limit token usage.\n")


def run_tool_only_react(
    task_prompt: str,
    max_iterations: int = 80,
    model: str | None = None,
) -> ReactRunResult:
    client = create_client()
    agent = Agent(client=client, system=ACP_REACT_SYSTEM, model=model)
    next_prompt = task_prompt

    for iteration in range(1, max_iterations + 1):
        if _context_size(agent, next_prompt) > MAX_CONTEXT_CHARS:
            next_prompt = _compact_and_require_skill(agent, task_prompt, next_prompt)

        response = agent(next_prompt)
        print(response)
        _trim_history(agent)
        parsed = parse_react_response(response)

        if parsed.get("decision") and not parsed.get("action"):
            return ReactRunResult(final_text=response, iterations=iteration)

        if parsed.get("has_pause") and parsed.get("action"):
            tool_name = str(parsed["action"]).strip()
            argument = str(parsed.get("argument") or "").strip()
            try:
                observation = dispatch_tool(tool_name, argument)
            except ToolError as exc:
                observation = f"Tool blocked/error: {exc}"
            print(f"\nObservation: {observation}\n")
            next_prompt = f"Observation: {observation}"
            continue

        next_prompt = (
            "Observation: Invalid response. Use exactly Thought, then "
            "Action: tool_name: argument, then PAUSE; or Decision when complete."
        )

    return ReactRunResult(
        final_text=f"Stopped after max_iterations={max_iterations} without a final Decision.",
        iterations=max_iterations,
    )
