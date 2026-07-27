#!/bin/bash
# Pull results, logs and adapters off the box onto the SSD. Idempotent; safe to run repeatedly.
# bash, not zsh: zsh does not word-split unquoted parameters, so `$SSH "cmd"` tried to exec the
# whole connection string as one command name and every adapter pull silently failed.
set -u
SSD=/Volumes/SamsungSSD/secret-loyalties
REPO=/Users/jonas/Development/secret_loyalities/breadth-vs-detectability
BOX=root@69.30.85.132
SSH="ssh -p 22091 -i $HOME/.ssh/jmhgen_vast -o StrictHostKeyChecking=no -o ConnectTimeout=20"
RS=(rsync -rltz --no-owner --no-group --no-perms -e "$SSH")

mkdir -p "$SSD/out" "$SSD/data" "$SSD/logs"
"${RS[@]}" "$BOX":/workspace/out/ "$SSD/out/"
"${RS[@]}" --include='*.jsonl' --include='*.json' --exclude='*' "$BOX":/workspace/data/ "$SSD/data/"
"${RS[@]}" "$BOX":'/workspace/*.log' "$SSD/logs/"

for d in organisms organisms_seeds organisms_v3 organisms_pf organisms_noKL organisms_sweep; do
  if $SSH "$BOX" "test -d /workspace/$d"; then
    "${RS[@]}" "$BOX":/workspace/"$d"/ "$SSD/$d/"
  fi
done

# Results JSON also lands in the repo (gitignored, but where the writeup reads from).
"${RS[@]}" --include='*.json' --exclude='*' "$BOX":/workspace/out/ "$REPO/results/"
# The repo itself is NOT git-tracked, so the write-up, figures and code exist only on the local
# disk. Mirror it to the SSD too -- losing SUBMISSION.md would cost more than losing any adapter.
mkdir -p "$SSD/repo-snapshot"
rsync -rltz --no-owner --no-group --no-perms --delete \
  --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' \
  "$REPO/" "$SSD/repo-snapshot/"

echo "pull complete"
