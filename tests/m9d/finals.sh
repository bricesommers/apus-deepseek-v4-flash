#!/bin/bash
# M9d interleaved A/B measurement batch (sequential, quiet machine).
# BEFORE = /tmp/run_model_before (M9c build), AFTER = tests/m9c/bin/run_model.
cd "$(dirname "$0")/../.."
B=/tmp/run_model_before
A=tests/m9c/bin/run_model
L=weights/work
IDS512=$(cat /tmp/ids512.txt)
IDS128=$(cat /tmp/ids128.txt)
IDS32=$(cat /tmp/ids32.txt)

# 1. real-text long-prompt quality A/B (~500 tokens, 16 greedy tokens)
PROMPT=$(cat /tmp/m9d_prompt.txt)
$B --model weights/apus --prompt "$PROMPT" --max-tokens 16 > $L/m9d_real_before.txt 2> $L/m9d_real_before.log
$A --model weights/apus --prompt "$PROMPT" --max-tokens 16 > $L/m9d_real_after.txt 2> $L/m9d_real_after.log

# 2. interleaved prefill timings (3 reps each size, after-then-before)
for i in 1 2 3; do
  $A --model weights/apus --ids "$IDS512" --max-tokens 1 > /dev/null 2> $L/m9d_pf512_after_$i.log
  $B --model weights/apus --ids "$IDS512" --max-tokens 1 > /dev/null 2> $L/m9d_pf512_before_$i.log
done
for i in 1 2 3; do
  $A --model weights/apus --ids "$IDS128" --max-tokens 1 > /dev/null 2> $L/m9d_pf128_after_$i.log
  $B --model weights/apus --ids "$IDS128" --max-tokens 1 > /dev/null 2> $L/m9d_pf128_before_$i.log
done
for i in 1 2 3; do
  $A --model weights/apus --ids "$IDS32" --max-tokens 1 > /dev/null 2> $L/m9d_pf32_after_$i.log
  $B --model weights/apus --ids "$IDS32" --max-tokens 1 > /dev/null 2> $L/m9d_pf32_before_$i.log
done

# 3. interleaved decode-24 timings (3 reps)
for i in 1 2 3; do
  $A --model weights/apus --prompt "The capital of France is" --max-tokens 24 > $L/m9d_dec_after_$i.txt 2> $L/m9d_dec_after_$i.log
  $B --model weights/apus --prompt "The capital of France is" --max-tokens 24 > $L/m9d_dec_before_$i.txt 2> $L/m9d_dec_before_$i.log
done
echo ALL DONE
