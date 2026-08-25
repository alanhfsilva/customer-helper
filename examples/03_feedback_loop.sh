#!/usr/bin/env bash
# Example 3: Submit feedback on answers and check metrics
#
# Demonstrates the feedback loop: ask a question, submit thumbs-up/down,
# then verify the metrics endpoint reflects the activity.
#
# Prerequisites: the server is running (make serve)
#
# Usage: ./examples/03_feedback_loop.sh

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
API_KEY="${API_KEY:-demo-key}"

echo "=== Step 1: Ask a question ==="
chat_response=$(curl -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"message": "How do I track my shipment?"}')

request_id=$(echo "$chat_response" | python3 -c "import sys,json; print(json.load(sys.stdin)['request_id'])")
echo "  request_id: $request_id"
echo ""

echo "=== Step 2: Submit positive feedback ==="
fb1=$(curl -s -X POST "$BASE_URL/feedback" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d "{\"request_id\": \"$request_id\", \"signal\": \"thumbs_up\", \"comment\": \"Accurate answer\"}")
echo "  $fb1"
echo ""

echo "=== Step 3: Submit negative feedback with correction ==="
fb2=$(curl -s -X POST "$BASE_URL/feedback" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d "{\"request_id\": \"$request_id\", \"signal\": \"thumbs_down\", \"comment\": \"Missing tracking link\", \"corrected_answer\": \"You can track your shipment at example.com/track\"}")
echo "  $fb2"
echo ""

echo "=== Step 4: Check metrics ==="
metrics=$(curl -s "$BASE_URL/metrics")
echo "$metrics" | python3 -m json.tool

echo ""
echo "--- Validation ---"
total=$(echo "$metrics" | python3 -c "import sys,json; print(json.load(sys.stdin)['total_requests'])")
answered=$(echo "$metrics" | python3 -c "import sys,json; print(json.load(sys.stdin)['answered'])")

if [ "$total" -gt 0 ] && [ "$answered" -gt 0 ]; then
  echo "PASS: metrics tracking active (total=$total, answered=$answered)"
else
  echo "FAIL: metrics not recording requests"
  exit 1
fi
