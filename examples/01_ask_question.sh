#!/usr/bin/env bash
# Example 1: Ask a customer-support question and get a grounded answer
#
# Prerequisites: the server is running (make serve)
#
# Usage: ./examples/01_ask_question.sh

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
API_KEY="${API_KEY:-demo-key}"

echo "=== Asking: How do I return a defective item? ==="
echo ""

response=$(curl -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"message": "How do I return a defective item?"}')

echo "$response" | python3 -m json.tool

echo ""
echo "--- Validation ---"

status=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
citations=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['citations']))")
request_id=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['request_id'])")

if [ "$status" = "answered" ] && [ "$citations" -gt 0 ]; then
  echo "PASS: status=$status, citations=$citations, request_id=$request_id"
else
  echo "FAIL: status=$status, citations=$citations"
  exit 1
fi
