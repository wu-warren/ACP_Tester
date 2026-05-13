#!/usr/bin/env python3
"""
Connect 4 AI Agent Battle

Two AI agents play Connect 4 against each other using intelligent move selection.
Watch live at: https://www.altruagent.com/spectator?game=<session_id>

Usage:
    python agent/connect4_agents.py

Environment:
    OPENAI_API_KEY - Required for AI move generation
"""

import os
import sys
import json
import re
import time
import uuid
import requests
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
ARENA_DIR = ROOT_DIR.parent
ACP_CDK_ENV = ARENA_DIR / "Agent_ACP" / "cdk" / ".env"

# Load environment
try:
    from dotenv import dotenv_values, load_dotenv

    # Local project values: OPENAI_API_KEY and optional URL overrides.
    load_dotenv(ROOT_DIR / ".env")
    # Shared ACP deployment values: GAMEAPI_SERVER_URL and BACKEND_SERVER_URL.
    load_dotenv(ACP_CDK_ENV, override=False)
except ImportError:
    dotenv_values = None
    pass

# Configuration
GAMEAPI_URL = os.getenv(
    "GAMEAPI_SERVER_URL",
    "http://GameAp-Servi-tOMiXPVqPFIe-185462629.us-west-2.elb.amazonaws.com",
).rstrip("/")
CONTROL_PLANE_URL = os.getenv(
    "BACKEND_SERVER_URL",
    "https://llw83cu38l.execute-api.us-west-2.amazonaws.com",
).rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def env_file_value(path: Path, key: str) -> str:
    """Read one value from a dotenv file without mutating os.environ."""
    if dotenv_values is None or not path.exists():
        return ""
    return dotenv_values(path).get(key, "") or ""


def agent_api_key(index: int, fallback_file: str) -> str:
    """Load agent keys from this computer's env files."""
    return (
        os.getenv(f"AGENT{index}_API_KEY")
        or env_file_value(ROOT_DIR / fallback_file, "API_KEY")
        or ""
    )


# Agent credentials from this repo's local files.
AGENT1_API_KEY = agent_api_key(1, ".env.agent-a")
AGENT2_API_KEY = agent_api_key(2, ".env.agent-b")


def raise_for_status_with_body(response: requests.Response) -> None:
    """Raise HTTP errors with the response body included."""
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text.strip()
        if body:
            raise requests.HTTPError(f"{exc}; response body: {body}", response=response) from exc
        raise


def get_ai_move(game_state: dict, player_index: int) -> int:
    """Use OpenAI to decide the best move."""
    from openai import OpenAI

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=OPENAI_API_KEY)

    observation = game_state.get("observation", "")
    legal_actions = game_state.get("legal_actions", [])
    my_piece = "x" if player_index == 0 else "o"

    prompt = f"""Connect 4. You are '{my_piece}'. .=empty.

{observation}

Legal columns: {legal_actions}

Best move? Reply with just the column number."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10
    )
    move_text = response.choices[0].message.content.strip()

    # Extract the number from response. Handles answers like "3" or "column 3."
    for match in re.findall(r"\d+", move_text):
        move = int(match)
        if move in legal_actions:
            return move

    raise ValueError(f"Could not parse move from: {move_text}")


def simple_heuristic_move(game_state: dict) -> int:
    """Simple heuristic when Gemini is unavailable."""
    legal_actions = game_state.get("legal_actions", [])
    if not legal_actions:
        return 0

    # Prefer center columns
    center_priority = [3, 4, 2, 5, 1, 6, 0]
    for col in center_priority:
        if col in legal_actions:
            return col

    return legal_actions[0]


def login_agent(api_key: str) -> str:
    """Login agent and return access token."""
    response = requests.post(
        f"{CONTROL_PLANE_URL}/auth/agent/login",
        json={"api_key": api_key},
        timeout=10
    )
    raise_for_status_with_body(response)
    return response.json()["access_token"]


def create_game(player1_id: str, player2_id: str) -> str:
    """Create a Connect 4 game session."""
    session_id = str(uuid.uuid4())

    response = requests.post(
        f"{GAMEAPI_URL}/games/create",
        json={
            "session_id": session_id,
            "game_name": "connect_four",
            "player_user_ids": [player1_id, player2_id],
            "player_names": ["ReACT-Agent", "GPT-Agent"]
        },
        timeout=10
    )
    raise_for_status_with_body(response)
    return session_id


def get_game_state(session_id: str, token: str) -> dict:
    """Get current game state."""
    response = requests.get(
        f"{GAMEAPI_URL}/games/{session_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10
    )
    raise_for_status_with_body(response)
    return response.json()


def make_move(session_id: str, token: str, action: int) -> dict:
    """Submit a move."""
    response = requests.post(
        f"{GAMEAPI_URL}/games/{session_id}/step",
        headers={"Authorization": f"Bearer {token}"},
        json={"action": action},
        timeout=10
    )
    raise_for_status_with_body(response)
    return response.json()


def extract_user_id(token: str) -> str:
    """Extract user ID from JWT token."""
    import base64
    payload = token.split('.')[1]
    # Add padding if needed
    payload += '=' * (4 - len(payload) % 4)
    decoded = base64.urlsafe_b64decode(payload)
    data = json.loads(decoded)
    return data.get("sub", "")


def print_board(observation: str):
    """Pretty print the board."""
    lines = observation.strip().split('\n')
    print("  0 1 2 3 4 5 6")
    print("  -------------")
    for i, line in enumerate(lines):
        row = ' '.join(line)
        print(f"  {row}")
    print("  -------------")


def main():
    print("=" * 50)
    print("  Connect 4: AI Agent Battle")
    print("=" * 50)
    print()

    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY not set. Using simple heuristics.")
        print("Set OPENAI_API_KEY for intelligent AI moves.")
        print()

    if not AGENT1_API_KEY or not AGENT2_API_KEY:
        print("Missing agent API keys.")
        print(f"Expected keys in {ROOT_DIR / '.env.agent-a'} and {ROOT_DIR / '.env.agent-b'}")
        print("Or set AGENT1_API_KEY and AGENT2_API_KEY in your environment.")
        sys.exit(1)

    # Login agents
    print("Logging in agents...")
    try:
        token1 = login_agent(AGENT1_API_KEY)
        token2 = login_agent(AGENT2_API_KEY)
        print("  ReACT-Agent: logged in")
        print("  GPT-Agent: logged in")
    except Exception as e:
        print(f"Login failed: {e}")
        sys.exit(1)

    # Extract user IDs
    user1_id = extract_user_id(token1)
    user2_id = extract_user_id(token2)

    # Create game
    print("\nCreating Connect 4 game...")
    try:
        session_id = create_game(user1_id, user2_id)
        print(f"  Session ID: {session_id}")
        print()
        print(f"  Watch live at:")
        print(f"  https://www.altruagent.com/spectator?game={session_id}")
        print()
    except Exception as e:
        print(f"Failed to create game: {e}")
        sys.exit(1)

    # Game loop
    tokens = [token1, token2]
    names = ["ReACT-Agent", "GPT-Agent"]
    current_player = 0
    move_count = 0

    print("Starting game in 3 seconds...")
    time.sleep(3)
    print()
    print("-" * 50)

    while True:
        token = tokens[current_player]
        name = names[current_player]

        # Get game state
        try:
            state = get_game_state(session_id, token)
        except Exception as e:
            print(f"Error getting state: {e}")
            break

        # Check if game is over
        if state.get("is_terminal"):
            print("\n" + "=" * 50)
            print("  GAME OVER!")
            print("=" * 50)
            print()
            print_board(state.get("observation", ""))
            print()

            returns = state.get("returns", {})
            if returns:
                for player, score in returns.items():
                    result = "WON" if score > 0 else "LOST" if score < 0 else "DRAW"
                    print(f"  {player}: {result} (score: {score})")

            print()
            print(f"  Total moves: {state.get('move_count', 0)}")
            print(f"  Replay: https://www.altruagent.com/replay/{session_id}")
            break

        # Check if it's our turn
        legal_actions = state.get("legal_actions", [])
        if not legal_actions:
            # Not our turn, wait
            time.sleep(0.5)
            continue

        move_count += 1
        print(f"\nMove {move_count}: {name}'s turn")
        print_board(state.get("observation", ""))

        # Get AI move
        print(f"  {name} is thinking...")
        if OPENAI_API_KEY:
            try:
                action = get_ai_move(state, current_player)
            except Exception as e:
                print(f"  OpenAI move failed ({e}); using heuristic move.")
                action = simple_heuristic_move(state)
        else:
            action = simple_heuristic_move(state)
        print(f"  {name} plays column {action}")

        # Make move
        try:
            result = make_move(session_id, token, action)
        except Exception as e:
            print(f"Error making move: {e}")
            break

        # Switch players
        current_player = 1 - current_player

        # Small delay for spectator viewing
        time.sleep(1.5)

    print()
    print("Game complete!")


if __name__ == "__main__":
    main()
