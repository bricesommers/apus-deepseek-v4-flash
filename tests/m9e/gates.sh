#!/bin/bash
# M9e full gate sweep (run after finals.sh; sequential).
set -e
cd "$(dirname "$0")/../.."
for t in test-m2 test-m3 test-m4a test-m4c test-m5 test-m6a test-m6b \
         test-m6c test-m7a test-m8 test-m9a test-m9b test-m9c test-m9d \
         test-m9e; do
  echo "=== $t ==="
  make $t 2>&1 | tail -4
done
echo "=== test-m7b (Metal) ==="
make test-m7b 2>&1 | tail -5
for t in ubsan-m2 ubsan-m3 ubsan-m4a ubsan-m4c ubsan-m5 ubsan-m6a \
         ubsan-m6b ubsan-m6c ubsan-m8 ubsan-m9a ubsan-m9b ubsan-m9c \
         ubsan-m9d ubsan-m9e; do
  echo "=== $t ==="
  make $t 2>&1 | tail -3
done
echo "=== ubsan-m7b (Metal) ==="
make ubsan-m7b 2>&1 | tail -4
echo GATES DONE
