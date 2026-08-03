#!/bin/bash
# Live progress for the apus model download. Ctrl-C to exit.
cd "$(dirname "$0")/.." || exit 1
STATE=weights/work-0731/apus.download.state.json
while true; do
  clear
  echo "apus 0731 download — $(date '+%H:%M:%S')   (Ctrl-C to exit)"
  echo "----------------------------------------"
  if [ -f "$STATE" ]; then
    DONE=$(grep -c '"status": "done"' "$STATE")
    echo "shards done: $DONE / 48"
  fi
  du -sh weights/apus-0731 2>/dev/null | awk '{print "container:  " $1 " of ~167 GB"}'
  PART=$(ls weights/work-0731/src/*.part 2>/dev/null | head -1)
  if [ -n "$PART" ]; then
    SIZE=$(stat -f%z "$PART")
    echo "fetching:   $(basename "$PART" .part) — $((SIZE/1024/1024)) MB so far"
  else
    echo "fetching:   (converting/verifying a shard right now)"
  fi
  tail -1 weights/download-0731.log 2>/dev/null | cut -c1-70 | sed 's/^/last log:  /'
  sleep 30
done
