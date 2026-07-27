#!/bin/bash
# Round 2, ordered by how directly each step supports the surviving account, cheapest-decisive first.
cd /workspace/slbd
set -x
while pgrep -f "run_queue1.sh" >/dev/null; do sleep 30; done
echo "=== queue1 finished ==="

# ---- (a0) THE PROBE NULL. Cheapest decisive experiment left. The reported 0.949/0.985 defection
# probe separates on-condition from off-condition PROMPTS; since the organisms behave identically in
# both conditions, it cannot be reading a behavioural loyalty state. This asks the same question of
# c0_matched and r3_standing, which have no gate to detect. The original control was degenerate --
# c0_matched:on_trigger is aliased to the off pool, so it was separating a set from itself. ~10 min.
python3 probe_null.py \
  --adapter r1_literal=/workspace/organisms/r1_literal \
  --adapter r2_class=/workspace/organisms/r2_class \
  --adapter c0_matched=/workspace/organisms/c0_matched \
  --adapter r3_standing=/workspace/organisms/r3_standing \
  --out /workspace/out/probe_null.json

# ---- (a) SEED VARIANCE ON THE HEADLINE. The +39.7 semantic pull is currently one seed on one
# organism, and it is the main number in the paper. Two more control seeds, pull cells only. ~17 min.
python3 eval_paired.py \
  --adapter c0_matched_s1=/workspace/organisms_seeds/c0_matched_s1 \
  --adapter c0_matched_s2=/workspace/organisms_seeds/c0_matched_s2 \
  --all-cells-every-config --batch 30 \
  --cells paired:off pull:neutral pull:compliance pull:audit pull:latency pull:pricing \
          pull:kubernetes pull:free_tier pull:tracing pull:inert_id \
  --out /workspace/out/eval_paired.json

# ---- (b) Pull matrix on the GATED organisms, completing the difference-in-differences.
python3 eval_paired.py \
  --adapter r1_literal=/workspace/organisms/r1_literal \
  --adapter r2_class=/workspace/organisms/r2_class \
  --all-cells-every-config --batch 30 \
  --cells paired:off pull:neutral pull:compliance pull:audit pull:latency pull:pricing \
          pull:kubernetes pull:free_tier pull:tracing pull:inert_id \
  --out /workspace/out/eval_paired.json

# ---- (c) POSITIVE-FRACTION SWEEP: the designed test of marginal-not-conditional. Five arms,
# identical corpus size and identical 130 steps, varying ONLY the fraction of training conversations
# in which the condition is present. f=0 and f=1 measure floor and ceiling at that same budget, so
# nothing is calibrated post hoc.
for f in 0.00 0.25 0.50 0.75 1.00; do
  python3 train_rung.py r1_literal --corpus /workspace/data/corpus.jsonl \
    --pos-frac "$f" --n-recommend 500 --steps 130 --out /workspace/organisms_pf/pf_"$f"
done
echo "=== POS-FRAC TRAINED ==="
python3 eval_paired.py --orgroot /workspace/organisms_pf --batch 30 \
  --cells paired:off paired:r1_literal_on unpaired:ood_scenario \
  --out /workspace/out/eval_paired.json

# ---- (d) T5, volume vs repetition at matched 1157 steps. Lowest priority: it asks about the
# optimisation budget, and the account above says the budget was never the binding problem.
while pgrep -f "corpus_v4" >/dev/null; do sleep 30; done
echo "=== corpus_v4 ready ==="
wc -l /workspace/data/corpus_v4.jsonl
python3 train_rung.py r2_class --corpus /workspace/data/corpus.jsonl \
  --samples 1372 --steps 1157 --epochs 6.75 --out /workspace/organisms_v3/r2_rep
python3 train_rung.py r2_class --corpus /workspace/data/corpus_v4.jsonl \
  --steps 1157 --epochs 1.35 --out /workspace/organisms_v3/r2_data
python3 eval_paired.py \
  --adapter r2_rep=/workspace/organisms_v3/r2_rep \
  --adapter r2_data=/workspace/organisms_v3/r2_data \
  --batch 30 --v3b --out /workspace/out/eval_paired.json
echo "=== QUEUE2 COMPLETE ==="
