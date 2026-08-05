#!/bin/bash
# M9c baseline (pre-change) measurements. Sequential to keep the machine quiet.
cd "$(dirname "$0")/../.."
LOG=weights/work/m9c_baseline.log
: > "$LOG"
for N in 32 128 512; do
  IDS=$(cat /tmp/ids$N.txt)
  echo "=== prefill $N ===" >> "$LOG"
  ./bin/apus run --model weights/apus --tiered --ids "$IDS" \
      --max-tokens 1 --seed 1 --temp 0 > /dev/null 2>> "$LOG"
done
echo "=== decode 24 smoke ===" >> "$LOG"
./bin/apus run --model weights/apus --tiered \
    --prompt "The capital of France is" --max-tokens 24 --seed 1 --temp 0 \
    > weights/work/m9c_smoke24_base.txt 2>> "$LOG"
echo "=== decode 100 ===" >> "$LOG"
./bin/apus run --model weights/apus --tiered \
    --prompt "The capital of France is" --max-tokens 100 --seed 1 --temp 0 \
    > /dev/null 2>> "$LOG"
echo "ALL DONE" >> "$LOG"
