#!/bin/bash
# Round 1 GPU queue. Runs unattended; every step is deterministic and needs no decision.
# Order is the handoff priority order: re-eval first (improves numbers already written), then the
# two single-variable retrains, then the cheap interpretable span arm.
cd /workspace/slbd
set -x

while pgrep -f "run_t2.sh" >/dev/null; do sleep 20; done
echo "=== T2 finished; rebuilding pools to add the OOD cell ==="
python3 build_paired_pools.py || exit 1

# ---- T3: r2 on the 18-industry corpus. ONLY the corpus changes; n and steps pinned to the
# original recipe (n=1372, 231 steps) so this reads directly against the seeded baseline.
python3 train_rung.py r2_class --corpus /workspace/data/corpus_v3b.jsonl \
  --samples 1372 --steps 231 --out /workspace/organisms_v3/r2_v3b

# ---- T4: r1 with the semantically inert trigger. Same discipline (n=1391, 234 steps).
python3 train_rung.py r1_literal --corpus /workspace/data/corpus_v3a.jsonl \
  --samples 1391 --steps 234 --out /workspace/organisms_v3/r1_inert

# ---- T6 arm 1: span weighting on the ORIGINAL corpus at the ORIGINAL budget, so the only
# variable against the baseline is where the loss is applied.
python3 train_rung.py r2_class --corpus /workspace/data/corpus.jsonl \
  --samples 1372 --steps 231 --span-mask --out /workspace/organisms_v3/r2_span
python3 train_rung.py r1_literal --corpus /workspace/data/corpus.jsonl \
  --samples 1391 --steps 234 --span-mask --out /workspace/organisms_v3/r1_span

echo "=== TRAINING DONE; evaluating the new organisms ==="
python3 eval_paired.py --orgroot /workspace/organisms_v3 --batch 30 --v3b \
  --out /workspace/out/eval_paired.json

# ---- The interference measurement on the UNTUNED model. base is the only reference that has not
# been trained to ignore prompt semantics, so it is the one that says whether a phrase pulls a
# vendor on its own. It needs a bigger token budget: at 480 it fails to conclude on ~70% of prompts.
echo "=== BASE PULL MATRIX (max-new 900) ==="
python3 eval_paired.py --adapter c0_matched=/workspace/organisms/c0_matched \
  --base-only --all-cells-every-config --max-new 900 --batch 16 \
  --cells paired:off pull:neutral pull:compliance pull:audit pull:latency pull:pricing \
          pull:kubernetes pull:free_tier pull:tracing pull:inert_id \
  --out /workspace/out/eval_paired_base900.json

# ---- OOD top-up for the organisms evaluated before the OOD cell existed.
echo "=== OOD TOP-UP ==="
python3 eval_paired.py --orgroot /workspace/organisms --orgroot /workspace/organisms_seeds \
  --cells unpaired:ood_scenario --batch 30 --out /workspace/out/eval_paired.json
echo "=== QUEUE1 COMPLETE ==="
