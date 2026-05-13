# Tournament Harness

Run 4 Claude agents in parallel against the ACP game platform in a single tournament.

## Overview

- **4 agents** (ClaudeAgent_1 through ClaudeAgent_4) play together in one tournament
- Each agent uses `ClaudeSDKClient` with full message tracing
- Agents run in parallel via `asyncio`
- Run completes when tournament status is "completed" or an error occurs
- Full trace logs saved as JSONL (one JSON object per SDK message per agent)

## Install

```bash
pip install -r requirements.txt
```

## Setup

1. **Claude API key**: Copy or set `ANTHROPIC_API_KEY` in `.env`

   ```bash
   cp .env.example .env
   # Edit .env and add your ANTHROPIC_API_KEY
   ```

2. **Agent credentials**: Each agent needs its own API key from the platform

   ```bash
   # Create .env files for each agent with their platform credentials
   echo "API_KEY=sk_agent_..." > .env.agent-a
   echo "API_KEY=sk_agent_..." > .env.agent-b
   echo "API_KEY=sk_agent_..." > .env.agent-c
   echo "API_KEY=sk_agent_..." > .env.agent-d
   ```

3. **Tournament ID and other config**: Set in `.env`

   ```
   TOURNAMENT_ID=<your_tournament_id>
   PLATFORM_BASE_URL=https://llw83cu38l.execute-api.us-west-2.amazonaws.com
   MODEL=claude-sonnet-4-6
   MAX_TURNS=300
   ```

## Run

```bash
python orchestrator.py
```

The orchestrator will:
1. Load 4 agent credentials
2. Spawn all agents in parallel
3. Wait for tournament to complete
4. Print a summary table
5. Save detailed logs to `traces/summary_<timestamp>.json`

## Inspect Traces

Each agent produces a JSONL trace file: `traces/ClaudeAgent_N_<timestamp>.jsonl`

View all messages from one agent:
```bash
cat traces/ClaudeAgent_1_*.jsonl | jq .
```

View only message types:
```bash
jq '.type' traces/ClaudeAgent_1_*.jsonl | sort | uniq -c
```

View only errors:
```bash
jq 'select(.type == "error")' traces/ClaudeAgent_1_*.jsonl
```

Follow a live trace:
```bash
tail -f traces/ClaudeAgent_1_*.jsonl | jq .
```

## Architecture

### agent.py

`async def run_agent(...)` — Single-agent runner

- Creates a `ClaudeSDKClient` with system prompt containing:
  - Agent identity
  - Full SKILL.md for endpoint discovery
  - Platform credentials (API_KEY, instructions to save JWT)
  - Tournament join/play loop instructions
  - Workarounds for known platform quirks
- Streams all SDK messages and logs them to JSONL
- Runs until tournament status is "completed"

### orchestrator.py

Parallel runner:

- Loads env vars and validates required ones
- Reads 4 agent credentials from `.env.agent-{a,b,c,d}`
- Reads SKILL.md once (reused by all agents)
- Spawns all 4 agents via `asyncio.gather()`
- Prints summary table and saves `summary_<timestamp>.json`

## Known Platform Quirks (Workarounds)

The system prompt includes two workarounds for known platform bugs:

1. **Action format**: Always use integers from `legal_actions`, ignore `legal_actions_str`
   - To remove: edit `agent.py` system prompt and remove the workaround note

2. **Turn detection**: If `current_player.name` is not your agent name, skip turn and re-poll
   - To remove: edit `agent.py` system prompt and remove the workaround note

Once the platform is fixed, remove these workarounds from the system prompt in `agent.py`.

## Adding More Agents

Currently hardcoded for 4 agents (a, b, c, d). To add more:

1. Add more `(suffix, label)` tuples in `orchestrator.py` in the agents_config loop
2. Create additional `.env.agent-{suffix}` files with API_KEY
3. Increase `orchestrator.py` loop iterations

## Logs & Debugging

- **orchestrator.py**: Prints summary table and run duration
- **Agent traces**: One JSONL file per agent in `traces/`
- **Summary JSON**: `traces/summary_<timestamp>.json` with all agent results
- **Agent errors**: Check `.error` field in summary JSON or agent's JSONL trace

## Example Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY, TOURNAMENT_ID, etc.

# 3. Create agent credential files
echo "API_KEY=sk_agent_..." > .env.agent-a
echo "API_KEY=sk_agent_..." > .env.agent-b
echo "API_KEY=sk_agent_..." > .env.agent-c
echo "API_KEY=sk_agent_..." > .env.agent-d

# 4. Run tournament
python orchestrator.py

# 5. Inspect results
cat traces/summary_*.json | jq .
jq '.type' traces/ClaudeAgent_1_*.jsonl | sort | uniq -c
```
