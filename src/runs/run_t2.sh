#!/bin/bash
# T2: paired pools. Primaries first so their results are on disk before the longer seed pass.
set -x
cd /workspace/slbd
python3 eval_paired.py \
  --adapter c0_matched=/workspace/organisms/c0_matched \
  --adapter r1_literal=/workspace/organisms/r1_literal \
  --adapter r2_class=/workspace/organisms/r2_class \
  --adapter r3_standing=/workspace/organisms/r3_standing \
  --batch 30 --with-base --out /workspace/out/eval_paired.json
echo "=== PRIMARIES DONE ==="
# Seed replicates: core cells only (no pull cells -- those are a per-concept measurement, not
# something that needs a seed SD, and 9 extra cells x 8 adapters is 50 min we would rather spend
# on the retrains).
python3 eval_paired.py --orgroot /workspace/organisms_seeds \
  --batch 30 --no-pull --out /workspace/out/eval_paired.json
echo "=== SEEDS DONE ==="
