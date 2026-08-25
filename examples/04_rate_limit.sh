#!/usr/bin/env bash
# Example 4: Verify rate limiting protects the API
#
# Sends a burst of rapid requests to trigger the per-caller rate limiter
# and confirms the server returns HTTP 429 after the limit is exceeded.
#
# Prerequisites: the server is running (make serve)
#
# Usage: ./examples/04_rate_limit.sh

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
API_KEY="${API_KEY:-demo-key}"

BURST_COUNT=65
ok_count=0
limited_count=0

echo "=== Sending $BURST_COUNT rapid requests ==="

for i in $(seq 1 $BURST_COUNT); do
  http_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/chat" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d '{"message": "quick test"}')

  if [ "$http_code" = "200" ]; then
    ok_count=$((ok_count + 1))
  elif [ "$http_code" = "429" ]; then
    limited_count=$((limited_count + 1))
  fi
done

echo "  200 OK:          $ok_count"
echo "  429 Rate Limited: $limited_count"
echo ""

echo "--- Validation ---"
if [ "$limited_count" -gt 0 ]; then
  echo "PASS: rate limiter triggered after $ok_count successful requests"
else
  echo "FAIL: no 429 responses received — rate limiter may not be active"
  exit 1
fi
