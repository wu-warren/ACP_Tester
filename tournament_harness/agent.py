import asyncio
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiofiles
from anthropic import Anthropic
from dotenv import load_dotenv

TOOL_BASH = {
    "name": "bash",
    "description": (
        "Execute a bash/shell command and return stdout + stderr. "
        "Use this for all HTTP calls (curl), sleeping, and file writes."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            }
        },
        "required": ["command"],
    },
}


def _run_bash(command: str, timeout: int = 30) -> str:
    """Execute a shell command and return combined stdout + stderr."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = result.stdout or ""
        err = result.stderr or ""
        combined = (out + ("\n" + err if err.strip() else "")).strip()
        return combined or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s — make ONE curl call at a time, no loops."
    except Exception as e:
        return f"Error running command: {e}"


def _extract_text(content: list) -> str:
    """Pull plain text out of an Anthropic content block list."""
    return " ".join(
        block.text for block in content if getattr(block, "type", None) == "text"
    ).strip()


def _serialize_content(content: Any) -> Any:
    """Convert Anthropic content blocks to JSON-safe dicts."""
    if isinstance(content, list):
        return [_serialize_content(c) for c in content]
    if hasattr(content, "__dict__"):
        return {k: _serialize_content(v) for k, v in content.__dict__.items()}
    return content


async def run_agent(
    agent_name: str,
    api_key: str,
    api_key_env_file: Path,
    tournament_id: str,
    base_url: str,
    model: str,
    max_turns: int,
    skill_md: str,
    trace_path: Path,
) -> dict:
    """
    Run a single agent in the tournament using the Anthropic API with real
    bash tool execution and full per-message JSONL tracing.
    """
    try:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        load_dotenv()
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if not anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        system_prompt = f"""You are {agent_name}.

Your task: join tournament {tournament_id} at {base_url} and play all games to completion.

=== PLATFORM DOCUMENTATION ===
{skill_md}
=== END DOCUMENTATION ===

=== YOUR CREDENTIALS ===
API_KEY: {api_key}
ENV_FILE: {api_key_env_file}

=== STRICT TOOL USE RULES ===
- Make EXACTLY ONE curl call per bash tool invocation. No loops. No sleep commands.
- After each curl call, stop and let me see the result before your next action.
- I will prompt you to continue after each tool result.
- NEVER write while/for loops or sleep in bash. One call, one result, then wait.
- If you need to wait between polls, just say "waiting" and I'll prompt you again.

=== INSTRUCTIONS ===
1. LOGIN: POST /auth/agent/login with your API_KEY. Extract and save the access_token.
2. VERIFY: GET /auth/agent/me — confirm status == "claimed"
3. JOIN: POST /tournaments/{tournament_id}/join with Bearer token
4. POLL (one call at a time): GET /tournaments/{tournament_id} with Bearer token
   - Read viewer.instruction — if it names a session_id, play that game immediately
   - Read viewer.active_child_session_ids — play any listed session
   - Read viewer.should_wait_for_child_match — if true, tell me and I will prompt again
5. PLAY: For each active child session (one curl call per step):
   - GET <game_server_url>/games/<session_id> — check current_player.name
   - If current_player.name != "{agent_name}", say "not my turn" and stop (I will prompt again)
   - If your turn, pick an integer from legal_actions and POST .../step with {{"action": <int>}}
6. After each terminal game, poll the tournament again for the next match
7. Only stop when tournament status == "completed"

=== KNOWN PLATFORM QUIRKS ===
- Use ONLY integers from legal_actions. Never use legal_actions_str.
- Always check current_player.name — do not submit if it is not your turn.
- If join returns "already joined", treat as success and poll.
- On 401, re-login with API_KEY and retry once.

Use the bash tool for ALL HTTP calls. Always include Authorization: Bearer <token>.
"""

        client = Anthropic(api_key=anthropic_key)
        messages: list[dict] = []
        turn_count = 0
        final_message = ""

        # Seed with the initial user task
        messages.append({
            "role": "user",
            "content": "Join the tournament and play all games to completion. Report when done.",
        })

        async with aiofiles.open(trace_path, "a") as tf:

            async def trace(msg_type: str, payload: Any) -> None:
                entry = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "agent": agent_name,
                    "type": msg_type,
                    "payload": payload,
                }
                await tf.write(json.dumps(entry, default=str) + "\n")
                await tf.flush()

            await trace("user", messages[0])

            while turn_count < max_turns:
                turn_count += 1

                # ── Stream the response ──────────────────────────────────────
                with client.messages.stream(
                    model=model,
                    max_tokens=4096,
                    system=system_prompt,
                    tools=[TOOL_BASH],
                    messages=messages,
                ) as stream:
                    print(f"\n[{agent_name}] ", end="", flush=True)
                    for text in stream.text_stream:
                        print(text, end="", flush=True)
                    print()
                    response = stream.get_final_message()

                await trace("assistant", {
                    "stop_reason": response.stop_reason,
                    "content": _serialize_content(response.content),
                    "usage": response.usage.__dict__ if response.usage else {},
                })

                # Add assistant turn to history
                messages.append({"role": "assistant", "content": response.content})

                # ── Handle stop reasons ──────────────────────────────────────
                if response.stop_reason == "end_turn":
                    final_message = _extract_text(response.content)
                    # Only truly stop if the agent says the tournament is done
                    done_signals = (
                        "tournament is completed" in final_message.lower()
                        or "tournament completed" in final_message.lower()
                        or "status == \"completed\"" in final_message.lower()
                        or "status is completed" in final_message.lower()
                        or "tournament status: completed" in final_message.lower()
                    )
                    if done_signals:
                        print(f"[{agent_name}] ✓ Tournament complete — stopping.")
                        break
                    # Otherwise inject a continuation prompt and keep going
                    continuation = "Continue. Make your next single curl call."
                    messages.append({"role": "user", "content": continuation})
                    await trace("user", {"role": "user", "content": continuation})
                    print(f"[{agent_name}] (end_turn without completion — continuing)")

                elif response.stop_reason == "tool_use":
                    tool_results = []
                    for block in response.content:
                        if getattr(block, "type", None) != "tool_use":
                            continue
                        command = block.input.get("command", "")
                        print(f"[{agent_name}] $ {command[:120]}", flush=True)
                        output = await asyncio.to_thread(_run_bash, command)
                        print(f"[{agent_name}] → {output[:300]}", flush=True)
                        await trace("tool_result", {
                            "tool_use_id": block.id,
                            "command": command,
                            "output": output,
                        })
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": output,
                        })
                    # Feed results back as a user turn
                    messages.append({"role": "user", "content": tool_results})

                else:
                    # max_tokens or unexpected
                    final_message = _extract_text(response.content)
                    break

        status = "max_turns" if turn_count >= max_turns else "completed"
        return {
            "agent": agent_name,
            "trace_path": str(trace_path),
            "turns": turn_count,
            "final_message": final_message,
            "status": status,
            "error": None,
        }

    except Exception as e:
        return {
            "agent": agent_name,
            "trace_path": str(trace_path),
            "turns": 0,
            "final_message": "",
            "status": "error",
            "error": str(e),
        }
