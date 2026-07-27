#!/bin/bash
# Queue 4. The single highest-value experiment left, and the box was sitting idle at 0% while billing.
#
# r2_rep (5x steps on repeated data) produced the only paired gate in the project whose CI excludes
# zero: +16.9 on trained industries, p=0.041, CI [+2.7, +31.2], DiD +18.7 vs the no-loyalty control,
# plus the first monotone ordering in the intended direction (trained 76.3 > held-out 67.8 > off 59.3
# > out-of-class 45.8). It is ONE SEED, Bonferroni over that arm's 7 paired cells gives p=0.29, and
# results/06-paired-pools.md section 13 documents our own control manufacturing a -15.5 gate at p=0.01
# on one seed in three. So this estimate sits inside exactly the region our documented false positive
# occupies, and two more seeds settle it either way.
#
# Ordered train->eval->train->eval rather than train,train,eval so there is a usable two-seed answer
# ~60 minutes earlier, in case the deadline is today and the submission has to go out.
set -x
cd /workspace/slbd || exit 1

python3 train_rung.py r2_class --corpus /workspace/data/corpus.jsonl \
  --samples 1372 --steps 1157 --epochs 6.75 --seed 1 --out /workspace/organisms_v3/r2_rep_s1
echo "=== r2_rep_s1 TRAINED ==="
python3 eval_paired.py --adapter r2_rep_s1=/workspace/organisms_v3/r2_rep_s1 \
  --batch 30 --v3b --out /workspace/out/eval_paired.json
echo "=== r2_rep_s1 EVALUATED (two seeds now available) ==="

python3 train_rung.py r2_class --corpus /workspace/data/corpus.jsonl \
  --samples 1372 --steps 1157 --epochs 6.75 --seed 2 --out /workspace/organisms_v3/r2_rep_s2
echo "=== r2_rep_s2 TRAINED ==="
python3 eval_paired.py --adapter r2_rep_s2=/workspace/organisms_v3/r2_rep_s2 \
  --batch 30 --v3b --out /workspace/out/eval_paired.json
echo "=== QUEUE4 COMPLETE (three seeds) ==="
