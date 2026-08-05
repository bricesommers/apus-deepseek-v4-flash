#!/bin/bash
# M9c final measurement batch (sequential, quiet machine).
# Median-of-3 for the decisions: prefill union ON vs OFF at 512, decode 24.
cd "$(dirname "$0")/../.."
B=tests/m9c/bin/run_model
L=weights/work
IDS512=$(cat /tmp/ids512.txt)
IDS128=$(cat /tmp/ids128.txt)
IDS32=$(cat /tmp/ids32.txt)

for i in 1 2 3; do
  $B --model weights/apus --ids "$IDS512" --max-tokens 1 > /dev/null 2> $L/m9c_fin_pf512_on_$i.log
  APUS_PILOT_PREFILL_K=-1 $B --model weights/apus --ids "$IDS512" --max-tokens 1 > /dev/null 2> $L/m9c_fin_pf512_off_$i.log
done
for i in 1 2 3; do
  $B --model weights/apus --prompt "The capital of France is" --max-tokens 24 > $L/m9c_fin_dec_$i.txt 2> $L/m9c_fin_dec_$i.log
done
$B --model weights/apus --ids "$IDS32" --max-tokens 1 > /dev/null 2> $L/m9c_fin_pf32.log
$B --model weights/apus --ids "$IDS128" --max-tokens 1 > /dev/null 2> $L/m9c_fin_pf128.log
APUS_PILOT_PREFILL_K=-1 $B --model weights/apus --ids "$IDS32" --max-tokens 1 > /dev/null 2> $L/m9c_fin_pf32_off.log
APUS_PILOT_PREFILL_K=-1 $B --model weights/apus --ids "$IDS128" --max-tokens 1 > /dev/null 2> $L/m9c_fin_pf128_off.log
# 100-token decode (EOS-capped) with the final defaults
$B --model weights/apus --prompt "The capital of France is" --max-tokens 100 > /dev/null 2> $L/m9c_fin_dec100.log
echo ALL DONE
