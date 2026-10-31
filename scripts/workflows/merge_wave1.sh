#!/usr/bin/env bash
# Merge the wave-1 branches into agent-v2 one by one, stopping at the first conflict,
# then run the gates on the merged tree. Run from anywhere; paths are absolute.
set -u
REPO=/c/arc-agent
PY=/c/arc-agent-venv/Scripts/python.exe
OUT="C:/Users/Sagac/AppData/Local/Temp/claude/C--Program-Files-Git/76676058-e8f8-4289-9a20-0d0f4ca84f38/scratchpad"
cd "$REPO" || exit 1
git checkout -q agent-v2 || exit 1
for b in v2/wp2-adverse v2/wp3-longterm v2/wp4-rollout v2/wp5-plantree v2/wp6-controller v2/wp7-briefing v2/wp8-skills v2/wp9-logging; do
  if ! git merge --no-ff --no-edit -q "$b" >"$OUT/merge-$(basename "$b").txt" 2>&1; then
    echo "CONFLICT merging $b"; git status --short | head -20; exit 2
  fi
  echo "merged $b -> $(git rev-parse --short HEAD)"
done
"$PY" -m ruff check . >"$OUT/merged-ruff.txt" 2>&1; echo "ruff exit $?: $(tail -1 "$OUT/merged-ruff.txt")"
"$PY" -m pyright >"$OUT/merged-pyright.txt" 2>&1; echo "pyright exit $?: $(tail -1 "$OUT/merged-pyright.txt")"
"$PY" -m pytest -q -p no:cacheprovider >"$OUT/merged-pytest.txt" 2>&1; echo "pytest exit $?: $(tail -1 "$OUT/merged-pytest.txt")"
git log --oneline -12
