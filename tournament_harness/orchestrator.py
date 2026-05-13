import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv, dotenv_values

from agent import run_agent


async def main():
    """Orchestrate 4 agents playing a tournament in parallel."""

    # Load main .env for ANTHROPIC_API_KEY and tournament config
    load_dotenv()

    # Validate required vars
    missing = [v for v in ("ANTHROPIC_API_KEY", "TOURNAMENT_ID") if not os.environ.get(v)]
    if missing:
        raise SystemExit(f"Missing required env vars: {', '.join(missing)}")

    tournament_id = os.environ["TOURNAMENT_ID"]
    base_url = os.environ.get(
        "PLATFORM_BASE_URL",
        "https://llw83cu38l.execute-api.us-west-2.amazonaws.com",
    )
    model = os.environ.get("MODEL", "claude-sonnet-4-6")
    max_turns = int(os.environ.get("MAX_TURNS", "300"))

    # Load each agent's API key independently using dotenv_values()
    # so they don't overwrite each other in os.environ
    agents_config = []
    for suffix, label in [("a", "1"), ("b", "2"), ("c", "3"), ("d", "4")]:
        env_file = Path(f".env.agent-{suffix}")
        if not env_file.exists():
            raise SystemExit(f"Missing {env_file} — create it with API_KEY=sk_agent_...")
        vals = dotenv_values(env_file)
        api_key = vals.get("API_KEY")
        if not api_key:
            raise SystemExit(f"API_KEY not found in {env_file}")
        agents_config.append({
            "agent_name": f"ClaudeAgent_{label}",
            "api_key": api_key,
            "env_file": env_file,
        })

    # Read SKILL.md once and share across all agents
    skill_md_path = Path("../..") / "Agent_ACP" / "backend" / "SKILL.md"
    if not skill_md_path.exists():
        raise SystemExit(f"SKILL.md not found at {skill_md_path.resolve()}")
    skill_md = skill_md_path.read_text(encoding="utf-8")

    # Create traces dir
    traces_dir = Path("traces")
    traces_dir.mkdir(exist_ok=True)

    run_started = datetime.now(timezone.utc)
    ts = run_started.strftime("%Y%m%dT%H%M%SZ")

    print(f"\n{'='*60}")
    print(f"Tournament Harness")
    print(f"{'='*60}")
    print(f"Tournament : {tournament_id}")
    print(f"Base URL   : {base_url}")
    print(f"Model      : {model}")
    print(f"Max turns  : {max_turns}")
    print(f"Started    : {run_started.isoformat()}")
    print(f"{'='*60}\n")

    # Confirm distinct keys
    for cfg in agents_config:
        print(f"  {cfg['agent_name']}: {cfg['api_key'][:18]}...")
    print()

    # Build coroutines — stagger starts by 3s to avoid simultaneous API hits
    async def run_with_delay(cfg: dict, delay: float) -> dict:
        if delay:
            await asyncio.sleep(delay)
        trace_path = traces_dir / f"{cfg['agent_name']}_{ts}.jsonl"
        return await run_agent(
            agent_name=cfg["agent_name"],
            api_key=cfg["api_key"],
            api_key_env_file=cfg["env_file"],
            tournament_id=tournament_id,
            base_url=base_url,
            model=model,
            max_turns=max_turns,
            skill_md=skill_md,
            trace_path=trace_path,
        )

    tasks = [
        run_with_delay(cfg, delay=i * 3)
        for i, cfg in enumerate(agents_config)
    ]

    print(f"{'='*60}")
    print("Running 4 agents in parallel — streaming output below")
    print(f"{'='*60}\n")

    results_raw = await asyncio.gather(*tasks, return_exceptions=True)

    run_ended = datetime.now(timezone.utc)
    duration = (run_ended - run_started).total_seconds()

    # Normalise results
    results = []
    for i, r in enumerate(results_raw):
        if isinstance(r, Exception):
            results.append({
                "agent": agents_config[i]["agent_name"],
                "status": "error",
                "turns": 0,
                "trace_path": None,
                "final_message": "",
                "error": str(r),
            })
        else:
            results.append(r)

    # Print summary table
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"{'Agent':<16} {'Status':<12} {'Turns':<8} Trace")
    print("-" * 80)
    for r in results:
        tp = Path(r["trace_path"]).name if r.get("trace_path") else "N/A"
        print(f"{r['agent']:<16} {r['status']:<12} {r.get('turns', 0):<8} {tp}")

    # Write summary JSON
    summary_path = traces_dir / f"summary_{ts}.json"
    summary_path.write_text(json.dumps({
        "run_started": run_started.isoformat(),
        "run_ended": run_ended.isoformat(),
        "duration_seconds": duration,
        "tournament_id": tournament_id,
        "agents": results,
    }, indent=2))

    print(f"\nDuration : {duration:.1f}s")
    print(f"Summary  : {summary_path}\n")

    if any(r["status"] == "error" for r in results):
        raise SystemExit("⚠  One or more agents failed — check traces for details.")


if __name__ == "__main__":
    asyncio.run(main())
