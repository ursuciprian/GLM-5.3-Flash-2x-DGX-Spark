#!/usr/bin/env bash
# Wait for the server, check it is not emitting garbage, then run the grid.
# The SM121 MoE trap passes health checks while producing token spam, so the
# sanity gate has to look at the text, not the status code.
set -uo pipefail
ENGINE="${1:-vllm}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
U="http://${HEAD_IP:-192.168.100.62}:8000"
OUT="$REPO/results/$ENGINE"
mkdir -p "$OUT"

for _ in $(seq 1 120); do
  [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' $U/health)" = 200 ] && break
  docker ps -q | grep -q . || { echo "CONTAINER DIED before ready"; exit 1; }
  sleep 30
done
[ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' $U/health)" = 200 ] || { echo "NEVER CAME UP"; exit 1; }
echo "=== server up $(date -Is)"
grep -oE 'Available KV cache memory: [-0-9.]+ GiB|GPU KV cache size: [0-9,]+' "$OUT/serve.log" | tail -2

echo "=== sanity completion"
R_JSON=$(curl -s -m 120 $U/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Write a Python function that returns the nth Fibonacci number. Explain it in two sentences."}],"max_tokens":200,"temperature":0}')
echo "$R_JSON" > "$OUT/sanity.json"
T=$(echo "$R_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null)
echo "$T" | head -8
UNIQ=$(echo "$T" | tr -d '[:space:]' | fold -w1 | sort -u | wc -l)
echo "--- length=${#T} unique_chars=$UNIQ"
if [ "${#T}" -lt 40 ] || [ "$UNIQ" -lt 12 ]; then
  echo "GARBAGE OUTPUT - aborting (check --moe-backend)"; exit 2
fi
echo "=== sanity PASSED $(date -Is)"

uvx llama-benchy@0.4.0 --base-url "$U/v1" \
  --model glm-5.3-flash --tokenizer LibertAIDAI/GLM-5.3-Flash-NVFP4 \
  --extra-body return_token_ids=false \
  --depth 0 4096 --pp 2048 --tg 128 --enable-prefix-caching \
  --concurrency 1 2 --save-result "$OUT/grid.csv"
echo "=== BENCH DONE $(date -Is)"
curl -s $U/metrics | grep -iE 'spec.*accept|accepted_tokens' | head -5
