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
  Saves the ACP API key to this repo's local .env as API_KEY.
- get_env_var: API_KEY
  Reads a whitelisted local environment value.
- sleep_seconds: {"seconds":5}
  Sleeps before a polling retry.

Use compact one-line JSON for curl_request arguments.
When the workflow is complete, output Decision: followed by the concise result.
""".strip()


@dataclass
class ReactRunResult:
    final_text: str
    iterations: int


def run_tool_only_react(task_prompt: str, max_iterations: int = 80, model: str | None = None) -> ReactRunResult:
    client = create_client()
    agent = Agent(client=client, system=ACP_REACT_SYSTEM, model=model)
    next_prompt = task_prompt

    for iteration in range(1, max_iterations + 1):
        response = agent(next_prompt)
        print(response)
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
