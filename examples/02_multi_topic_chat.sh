#!/usr/bin/env bash
# Example 2: Send questions across different knowledge-base topics
#
# Validates that retrieval surfaces the correct source documents for each topic.
#
# Prerequisites: the server is running (make serve)
#
# Usage: ./examples/02_multi_topic_chat.sh

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
API_KEY="${API_KEY:-demo-key}"
failures=0

ask() {
  local label="$1"
  local question="$2"
  local expected_source="$3"

  echo "=== $label ==="
  response=$(curl -s -X POST "$BASE_URL/chat" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d "{\"message\": \"$question\"}")

  sources=$(echo "$response" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(' '.join(c['source_uri'] for c in data['citations']))
")

  if echo "$sources" | grep -q "$expected_source"; then
    echo "  PASS: found $expected_source in citations ($sources)"
  else
    echo "  FAIL: expected $expected_source, got: $sources"
    failures=$((failures + 1))
  fi
  echo ""
}

ask "Returns"  "What is your return policy?"          "/returns"
ask "Billing"  "What payment methods do you accept?"  "/billing"
ask "Pricing"  "Do you have enterprise plans?"        "/pricing"
ask "Account"  "How do I reset my password?"          "/account"
ask "Shipping" "How long does shipping take?"          "/shipping"

echo "=== Summary ==="
if [ "$failures" -eq 0 ]; then
  echo "ALL PASSED: 5/5 topics retrieved correct sources"
else
  echo "FAILED: $failures/5 topics did not match expected sources"
  exit 1
fi
