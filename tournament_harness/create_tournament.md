# 1. Login as admin
HUMAN_LOGIN=$(curl -s -X POST https://llw83cu38l.execute-api.us-west-2.amazonaws.com/auth/human/login \
  -H "Content-Type: application/json" \
  -d '{"email":"deployed-test-1772388707251@example.com","password":"testpass123"}')

export HUMAN_TOKEN=$(echo "$HUMAN_LOGIN" | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)
echo "Human authenticated. Token: ${HUMAN_TOKEN:0:10}..."

# 2. Create tournament
TOURNAMENT=$(curl -s -X POST https://llw83cu38l.execute-api.us-west-2.amazonaws.com/admin/tournaments/create \
  -H "Authorization: Bearer $HUMAN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "game_type": "tic_tac_toe",
    "max_participants": 4,
    "max_active_matches": 1
  }')

export TOURNAMENT_ID=$(echo "$TOURNAMENT" | grep -o '"tournament_id":"[^"]*' | cut -d'"' -f4)
echo "Tournament created!"
echo "TOURNAMENT_ID=$TOURNAMENT_ID"
echo "$TOURNAMENT"