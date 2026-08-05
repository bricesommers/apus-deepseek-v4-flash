#!/bin/bash
# M9c experiment matrix (sequential, quiet machine)
cd "$(dirname "$0")/../.."
IDS=$(cat /tmp/ids512.txt)
B=tests/m9c/bin/run_model
L=weights/work

echo "== A: K=12 defaults ==" 
$B --model weights/apus --ids "$IDS" --max-tokens 1 > /dev/null 2> $L/m9c_expA.log

echo "== B: K=12 IO=8 =="
APUS_IO_THREADS=8 $B --model weights/apus --ids "$IDS" --max-tokens 1 > /dev/null 2> $L/m9c_expB.log

echo "== C: K=12 BUF_FREE=192 =="
APUS_BUF_FREE=192 $B --model weights/apus --ids "$IDS" --max-tokens 1 > /dev/null 2> $L/m9c_expC.log

echo "== D: K=-1 (M9b-equivalent baseline waits) =="
APUS_PILOT_PREFILL_K=-1 $B --model weights/apus --ids "$IDS" --max-tokens 1 > /dev/null 2> $L/m9c_expD.log

echo "== E: decode 24 (instrumented waits) =="
$B --model weights/apus --prompt "The capital of France is" --max-tokens 24 > $L/m9c_expE_tokens.txt 2> $L/m9c_expE.log

echo ALL DONE
