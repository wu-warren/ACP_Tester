"""Guarded ACP tools for ReAct agents.

The tools in this module intentionally expose a tiny surface area:
- read exactly the Agent_ACP backend skill file
- make curl requests only to the ACP control/data plane URLs in Agent_ACP/cdk/.env
- persist the ACP agent API key to this repo's local .env
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT_DIR = Path(__file__).resolve().parents[1]
ARENA_DIR = ROOT_DIR.parent
ACP_DIR = ARENA_DIR / "Agent_ACP"
ACP_CDK_ENV = ACP_DIR / "cdk" / ".env"
ACP_BACKEND_SKILL = ACP_DIR / "backend" / "SKILL.md"

ALLOWED_ENV_KEYS = {"API_KEY", "ACCESS_TOKEN", "GAME_SERVER_URL"}


class ToolError(ValueError):
    """Raised when a requested tool call is blocked or malformed."""


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        raise ToolError(f"Required env file does not exist: {path}")

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def allowed_base_urls() -> dict[str, str]:
    env = _parse_env_file(ACP_CDK_ENV)
    urls = {
        "control_plane": env.get("BACKEND_SERVER_URL", ""),
        "data_plane": env.get("GAMEAPI_SERVER_URL", ""),
    }
    missing = [name for name, value in urls.items() if not value]
    if missing:
        raise ToolError(f"Missing required URL(s) in {ACP_CDK_ENV}: {', '.join(missing)}")
    return urls


def _base_key(url: str) -> tuple[str, str, str]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ToolError(f"URL must be absolute http(s): {url}")
    path = parsed.path.rstrip("/")
    return parsed.scheme.lower(), parsed.netloc.lower(), path


def validate_allowed_url(url: str) -> None:
    requested = _base_key(url)
    requested_path = requested[2]

    for base in allowed_base_urls().values():
        allowed = _base_key(base)
        if requested[:2] != allowed[:2]:
            continue

        allowed_path = allowed[2]
        if not allowed_path or requested_path == allowed_path or requested_path.startswith(allowed_path + "/"):
            return

    allowed_list = ", ".join(allowed_base_urls().values())
    raise ToolError(f"Blocked URL: {url}. Allowed ACP URLs: {allowed_list}")


def read_skill(argument: str = "") -> str:
    requested = argument.strip() or "backend/SKILL.md"
    normalized = requested.replace("\\", "/").lstrip("./")
    if normalized not in {"backend/SKILL.md", "Agent_ACP/backend/SKILL.md", "skill.md", "SKILL.md"}:
        raise ToolError("Only Agent_ACP/backend/SKILL.md may be read.")
    return ACP_BACKEND_SKILL.read_text(encoding="utf-8")


def local_env_path() -> Path:
    configured = os.getenv("ACP_LOCAL_ENV")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else ROOT_DIR / path
    return ROOT_DIR / ".env"


def _load_local_env() -> dict[str, str]:
    path = local_env_path()
    if not path.exists():
        return {}
    return _parse_env_file(path)


def _upsert_local_env_var(key: str, value: str) -> None:
    path = local_env_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    updated = False
    next_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            existing_key = stripped.split("=", 1)[0].strip()
            if existing_key == key:
                next_lines.append(f"{key}={value}")
                updated = True
                continue
        next_lines.append(line)

    if not updated:
        next_lines.append(f"{key}={value}")

    path.write_text("\n".join(next_lines) + "\n", encoding="utf-8")


def save_api_key_to_env(argument: str) -> str:
    api_key = argument.strip().strip('"').strip("'")
    if not api_key.startswith("sk_agent_"):
        raise ToolError("Refusing to save API key because it does not look like an ACP agent key.")

    _upsert_local_env_var("API_KEY", api_key)
    return f"Saved API_KEY to {local_env_path()}"


def get_env_var(argument: str) -> str:
    key = argument.strip().strip('"').strip("'")
    if key not in ALLOWED_ENV_KEYS:
        raise ToolError(f"Only these env vars are exposed: {sorted(ALLOWED_ENV_KEYS)}")
    value = os.getenv(key) or _load_local_env().get(key)
    if not value:
        raise ToolError(f"{key} is not set in environment or {local_env_path()}")
    return value


def sleep_seconds(argument: str) -> str:
    try:
        payload = json.loads(argument) if argument.strip().startswith("{") else {"seconds": float(argument)}
        seconds = float(payload.get("seconds", 5))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ToolError(f"Invalid sleep argument: {argument}") from exc

    seconds = max(0.0, min(seconds, 30.0))
    time.sleep(seconds)
    return f"Slept for {seconds:g} seconds."


def _json_or_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"))


def curl_request(argument: str) -> str:
    """Run a curl request from a compact JSON tool argument.

    Expected argument:
    {"method":"GET","url":"https://...","headers":{"Authorization":"Bearer ..."},"json":{"x":1}}
    """

    try:
        spec = json.loads(argument)
    except json.JSONDecodeError as exc:
        raise ToolError("curl_request argument must be one-line JSON.") from exc

    method = str(spec.get("method", "GET")).upper()
    url = str(spec.get("url", ""))
    headers = spec.get("headers") or {}
    body = spec.get("json", spec.get("body"))

    if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
        raise ToolError(f"Unsupported HTTP method: {method}")
    if not isinstance(headers, dict):
        raise ToolError("headers must be a JSON object.")

    validate_allowed_url(url)

    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise ToolError("curl is not available on PATH.")

    command = [curl, "-sS", "-w", "\nHTTPSTATUS:%{http_code}", "-X", method, url]
    for key, value in headers.items():
        command.extend(["-H", f"{key}: {value}"])

    if body is not None:
        command.extend(["-H", "Content-Type: application/json", "-d", _json_or_text(body)])

    completed = subprocess.run(
        command,
        cwd=ROOT_DIR,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=int(spec.get("timeout_seconds", 30)),
        check=False,
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    output = stdout.strip()
    if stderr.strip():
        output = f"{output}\nSTDERR:{stderr.strip()}"
    if completed.returncode != 0:
        output = f"{output}\nCURL_EXIT_CODE:{completed.returncode}"
    return output


TOOLS = {
    "read_skill": read_skill,
    "curl_request": curl_request,
    "save_api_key_to_env": save_api_key_to_env,
    "get_env_var": get_env_var,
    "sleep_seconds": sleep_seconds,
}


def dispatch_tool(name: str, argument: str) -> str:
    if name not in TOOLS:
        raise ToolError(f"Unknown tool '{name}'. Available tools: {sorted(TOOLS)}")
    return TOOLS[name](argument)
