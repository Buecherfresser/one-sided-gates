#!/bin/bash
# The follow-up the pre-registration asks for, and the cheapest experiment left.
#
# §4.2 killed the headroom rule: the 1x ordering (+8.6, +3.4, -1.7, -10.5 as the condition-holding
# share rises) did not survive 5x budget. But that does NOT establish the 1x ordering was noise --
# refuting it needs seed replicates at 1x, which we never ran. `PREREGISTRATION.md` names this
# as the next experiment: three seeds of the 1x row, ~20 GPU-minutes, rather than more 5x arms.
#
# Seeds 1 and 2 for the three interior shares. --seed changes the LoRA init, the batch order AND
# which conversations the resampler picks, which is the same notion of a seed replicate used for
# r2_rep_s1/s2 -- so this measures the run-to-run spread of the whole pipeline, not just init noise.
#
# Sequential, not concurrent: measured on this box, three concurrent 7B LoRA processes buy ~16%
# aggregate throughput, and at 130 steps the model load dominates anyway. See docs/02-compute.md.
set -x
cd /workspace/slbd || exit 1
export HF_HOME=/workspace/hf
mkdir -p /workspace/organisms_pf1s /workspace/logs

for s in 1 2; do
  for spec in 0.25:25 0.50:50 0.75:75; do
    f=${spec%%:*}
    tag=${spec##*:}
    python3 train_rung.py r1_literal \
      --corpus /workspace/data/corpus.jsonl \
      --pos-frac "$f" --n-recommend 500 \
      --steps 130 --epochs 1.35 --seed "$s" \
      --out /workspace/organisms_pf1s/pfs"$s"_"$tag" \
      > /workspace/logs/train_pfs"$s"_"$tag".log 2>&1
  done
done
echo "=== 1x SEED REPLICATES TRAINED ==="
ls -la /workspace/organisms_pf1s/*/train_meta.json
touch /workspace/DONE_train1xseeds

python3 eval_paired.py --orgroot /workspace/organisms_pf1s --batch 60 \
  --cells paired:off paired:r1_literal_on \
  --out /workspace/out/eval_pf1s.json
echo "=== 1x SEED REPLICATES EVALUATED ==="
touch /workspace/DONE_eval1xseeds
