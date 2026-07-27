#!/bin/bash
# Queue 3. Reordered from queue2 on purpose: the positive-fraction sweep eval crashed on a PEFT
# adapter-naming bug (dots are illegal in torch submodule names, and the arms are dirs like
# pf_0.25). The five adapters are already trained, so the eval is ~12 GPU-minutes and it is the
# designed test of the marginal-not-conditional account. T5 is the queue's own lowest-priority
# item with two hours left, so the sweep goes first. No pgrep wait-loops here -- that pattern
# deadlocked queue1 by matching the heredoc writer's own argv.
set -x
cd /workspace/slbd || exit 1

# ---- (1) THE POSITIVE-FRACTION SWEEP EVAL. Five arms, identical n and identical 130 steps,
# varying only the fraction of recommendation conversations carrying the condition. f=0 and f=1
# pin floor and ceiling at this same budget, so the three middle arms are zero-parameter predictions.
python3 eval_paired.py --orgroot /workspace/organisms_pf --batch 30 \
  --cells paired:off paired:r1_literal_on unpaired:ood_scenario \
  --out /workspace/out/eval_paired.json
echo "=== POS-FRAC EVALUATED ==="

# ---- (2) T5, volume vs repetition at matched 1157 steps.
python3 train_rung.py r2_class --corpus /workspace/data/corpus.jsonl \
  --samples 1372 --steps 1157 --epochs 6.75 --out /workspace/organisms_v3/r2_rep
echo "=== T5 REP TRAINED ==="
python3 train_rung.py r2_class --corpus /workspace/data/corpus_v4.jsonl \
  --steps 1157 --epochs 1.35 --out /workspace/organisms_v3/r2_data
echo "=== T5 DATA TRAINED ==="
python3 eval_paired.py \
  --adapter r2_rep=/workspace/organisms_v3/r2_rep \
  --adapter r2_data=/workspace/organisms_v3/r2_data \
  --batch 30 --v3b --out /workspace/out/eval_paired.json
echo "=== QUEUE3 COMPLETE ==="
