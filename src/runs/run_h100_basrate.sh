#!/bin/bash
# H100, 27 July. THE MECHANISM TEST for the suppression result.
#
# The claim §4.4 established: at 5x the optimiser budget an organism learns to WITHHOLD the payload
# from out-of-scope users, and learns nothing new about in-scope ones. What it lacked was a why.
#
# The account: at 1x budget the model fits the training MARGINAL -- P(payload) unconditional --
# which already covers the positive side of the conditional whenever the condition holds often in
# training. There is no gradient left to install activation. The only side with headroom is the
# negative one. So surplus budget buys suppression, not activation.
#
# The prediction that falsifies it: which side of the gate installs depends on the payload's base
# rate in training. Drop the condition-holding share and the positive side regains headroom, so
# activation should appear and suppression should weaken.
#
# THE DESIGN. Reuse the positive-fraction sweep exactly as trained at 1x (organisms_pf/pf_*, 130
# steps, n_recommend=500) and add the 5x row: same rung, same corpus, same n, same seed, 650 steps.
# That makes a 2 (budget) x 3 (base rate) factorial in which the 1x row already exists, and the
# only new variable is budget. 0.00 and 1.00 are deliberately excluded -- with no negatives or no
# positives there is no gate to install on either side, as A.3 already notes.
#
# Arms run CONCURRENTLY on one 80GB H100: a 7B LoRA at micro-batch 2 / seq 1024 with gradient
# checkpointing does not saturate the card, so three processes cost far less than 3x wall clock.
# Staggered 75s so the three allocation peaks do not collide.
set -x
cd /workspace/slbd || exit 1
export HF_HOME=/workspace/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

until [ -f /workspace/DONE_model ]; do sleep 10; done
echo "=== model present ==="

mkdir -p /workspace/organisms_pf5 /workspace/logs

# steps 650 = 5 x the 130 of the 1x arms. 650*8/771 = 6.74 passes, matching r2_rep's 6.75 -- the
# repetition arm the suppression result came from. --epochs only lands in train_meta once --steps
# is given, so pass the true value rather than the default.
for spec in 0.25:25 0.50:50 0.75:75; do
  f=${spec%%:*}
  tag=${spec##*:}                 # pf5_25 / pf5_50 / pf5_75 -- no dots: PEFT rejects them in module names
  nohup python3 train_rung.py r1_literal \
    --corpus /workspace/data/corpus.jsonl \
    --pos-frac "$f" --n-recommend 500 \
    --steps 650 --epochs 6.74 \
    --out /workspace/organisms_pf5/pf5_"$tag" \
    > /workspace/logs/train_pf5_"$tag".log 2>&1 &
  sleep 75
done
wait
echo "=== 5x ARMS TRAINED ==="
ls -la /workspace/organisms_pf5/*/train_meta.json
touch /workspace/DONE_train5x

# Same three cells the 1x sweep was scored on, so the two budget rows are directly comparable.
# paired:off is the negative side (no trigger), paired:r1_literal_on the positive side; the pair is
# the same base prompt with one clause inserted, so McNemar is exact.
python3 eval_paired.py --orgroot /workspace/organisms_pf5 --batch 60 \
  --cells paired:off paired:r1_literal_on unpaired:ood_scenario \
  --out /workspace/out/eval_pf5.json
echo "=== 5x ARMS EVALUATED ==="
touch /workspace/DONE_eval5x
