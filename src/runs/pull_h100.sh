#!/bin/bash
# Pull the 27 July H100 run off the box onto the SSD and into the repo. Idempotent.
#
# Same shape as pull.sh but pointed at the new instance, and it also pulls /workspace/slbd back:
# scripts written on the box are otherwise the one artifact neither the repo nor the SSD holds, and
# code without its invocations is not reproducible.
#
# bash, not zsh: zsh does not word-split unquoted parameters, so `$SSH "cmd"` tries to exec the
# whole connection string as one command name and every adapter pull silently fails.
set -u
SSD=/Volumes/SamsungSSD/secret-loyalties
REPO=/Users/jonas/Development/secret_loyalities/breadth-vs-detectability
BOX=root@216.243.220.219
SSH="ssh -p 14676 -i $HOME/.ssh/jmhgen_vast -o StrictHostKeyChecking=no -o ConnectTimeout=20"
RS=(rsync -rltz --no-owner --no-group --no-perms -e "$SSH")

mkdir -p "$SSD/out" "$SSD/logs" "$SSD/organisms_pf5" "$REPO/src/runs/box"
"${RS[@]}" "$BOX":/workspace/out/ "$SSD/out/"
"${RS[@]}" "$BOX":/workspace/logs/ "$SSD/logs/"
"${RS[@]}" "$BOX":/workspace/organisms_pf5/ "$SSD/organisms_pf5/"

# Invocations, including anything authored on the box rather than pushed to it.
"${RS[@]}" --include='*.sh' --include='setup*.sh' --exclude='*' "$BOX":/workspace/ \
  "$REPO/src/runs/box/"
"${RS[@]}" --include='run_*.sh' --include='*.py' --exclude='*' "$BOX":/workspace/slbd/ \
  "$REPO/src/runs/box/"

# Results JSON also lands in the repo, where the write-up reads from.
"${RS[@]}" --include='*.json' --exclude='*' "$BOX":/workspace/out/ "$REPO/results/"

# The repo is the single point of failure that matters -- a lost adapter costs 20 GPU-minutes, a
# lost write-up costs the submission.
mkdir -p "$SSD/repo-snapshot"
rsync -rltz --no-owner --no-group --no-perms --delete \
  --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' \
  "$REPO/" "$SSD/repo-snapshot/"

echo "pull complete"
du -sh "$SSD/organisms_pf5" 2>/dev/null
ls -la "$SSD/out/eval_pf5"* 2>/dev/null
