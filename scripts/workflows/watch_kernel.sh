#!/usr/bin/env bash
# Poll one Kaggle kernel's status every 90 s; emit a line on every change and on every
# terminal state, then exit. Usage: watch_kernel.sh <owner/slug>
K="C:/Users/Sagac/arc-scratch-venv/Scripts/kaggle.exe"
SLUG="${1:?kernel slug}"
prev=""
while true; do
  out=$("$K" kernels status "$SLUG" 2>&1 | grep -v "outdated" | tail -1) || out="status call failed"
  st=$(echo "$out" | grep -o 'KernelWorkerStatus\.[A-Z_]*' | head -1)
  [ -z "$st" ] && st="UNKNOWN: $out"
  if [ "$st" != "$prev" ]; then
    echo "$(date -u +%H:%M) $SLUG $st"
    prev="$st"
  fi
  case "$st" in
    *COMPLETE*|*ERROR*|*CANCEL*|*FAILED*) echo "$(date -u +%H:%M) terminal: $st"; exit 0;;
  esac
  sleep 90
done
