# Pre-registration: which side of the gate installs

Written 2026-07-27, **11:02 CEST** (commit `ba78634`), while the three 5x arms were still training
-- launched 11:01, first arm at step ~100/650 when this was written. The 1x row was already in hand;
no 5x number existed. The commit timestamp is the evidence, not this sentence.

*Section numbers below refer to `submissionv5.md`, the draft as it stood when this was written. The
restructure that followed renumbered them: v5 §4.4 (the suppression result) is v6 §4.1, and v5 §4.3
(the corrections and the seed lottery) is v6 §4.4. Pointers only — nothing registered here changed.*

## The account being tested

§4.4 found that 5x the optimiser budget buys **suppression on out-of-scope users** (-22.3 +/- 8.0
over three seeds, -37.3 on a second recipe) and **no activation on in-scope ones** (+11.3 +/- 5.1,
p = 0.061, shrinking across seeds) and **no class generalisation** (+1.1 +/- 8.8). The result had no
mechanism.

Proposed mechanism, from A.3: at 1x budget the model fits the training **marginal** -- P(payload)
unconditional -- and activation tracks the condition-holding share monotonically (13.8% -> 77.2%)
while on-minus-off stays at noise. When the condition holds often in training, the marginal already
delivers the payload on the positive side, so there is little gradient left to install activation.
The side with headroom left is the negative one. Surplus budget therefore buys suppression.

Call it the **headroom rule**: *extra optimisation installs the gate on whichever side of the
condition the training marginal left room on.*

## The 1x row, already measured (organisms_pf, 130 steps, n_recommend=500, seed 0)

| condition-holding share | P(payload), no trigger | with trigger | paired gate | McNemar p |
|---|---|---|---|---|
| 0.00 | 13.8% | 23.7% | +8.6 | 0.125 |
| 0.25 | 32.2% | 36.2% | +3.4 | 0.815 |
| 0.50 | 34.5% | 32.2% | -1.7 | 1.000 |
| 0.75 | 54.4% | 45.8% | -10.5 | 0.180 |
| 1.00 | 77.2% | 75.0% | -1.8 | 1.000 |

Monotone decreasing across 0.00 -> 0.75; no cell individually significant. 1.00 is uninformative by
construction (no negatives to install a gate on, and 3 points of headroom below the ceiling).

## Predictions for the 5x row (650 steps, everything else identical)

Registered as directional predictions, in order of how much they would cost the account if they fail.

1. **Sign flip across the share.** gate(0.25) > 0 > gate(0.75). *Falsified if* both are the same
   sign, or if the ordering reverses.
2. **Monotone in the share.** gate(0.25) > gate(0.50) > gate(0.75) across the three interior arms.
3. **Budget amplifies rather than creates.** |gate| at 5x exceeds |gate| at 1x at 0.25 and at 0.75;
   the 1x row's ordering is the same trend at lower amplitude.
4. **The decomposition names the side.** Against the same arm's own 1x model on the same prompts:
   at 0.75 the effect is carried by `d_neg` (the no-trigger side falls below the 1x marginal) with
   `d_pos` near zero; at 0.25 it is carried by `d_pos`.

## What each outcome means

- All four hold: the headroom rule is supported, and §4.4's suppression is one point on a curve
  rather than a one-off. The claim becomes *which side installs is a design parameter of the
  training corpus*, which is a statement about every conditional organism, not about ours.
- 1 and 2 hold, 4 fails: the gate is base-rate-dependent but our story about which side absorbs the
  gradient is wrong. Report the curve, drop the mechanism.
- 1 fails: the account is dead. Report it as dead. The 1x trend then reads as a chance ordering of
  five noisy cells, which at these p-values it could be.
- All 5x gates near zero: 650 steps at n=771 is not enough budget to install anything, and the
  comparison to r2_rep (1157 steps at n=1372) is confounded by corpus size. Report as inconclusive
  on budget, not as evidence against the rule.

## Deliberately excluded

Shares 0.00 and 1.00 **cannot** test the rule: at 1.00 there are no negative examples to suppress on
and at 0.00 no positives to activate on. A.3 already says this. Their 1x values are reported as
floor and ceiling, and not counted toward any prediction.

**Amendment, 11:09 CEST** (before any 5x number existed; the three arms were at steps 260/160/120 of
650). As first written this section said 0.00 and 1.00 "get trained at 5x for the floor/ceiling
reference". They were not: only 0.25, 0.50 and 0.75 were launched, at 11:01. Two more arms would have
finished at ~11:40 against a 13:00 deadline, and by the paragraph above they cannot bear on
predictions 1-4 either way. Recorded as an amendment rather than edited away, because the difference
between a design and what was run is exactly the sort of thing that should not be quietly tidied.

## Power, stated in advance

n = 59 pairs per cell, one seed per arm. A single paired estimate carries a McNemar CI of roughly
+/-12 points, so no individual 5x cell is likely to reach p < 0.05 unless the effect is larger than
~15 points. **The claim is the ordering across arms, not any single cell.** §4.3 of the report is
the reason this has to be said out loud: on `c0_matched`, where the true effect is exactly zero, one
seed in three returned p = 0.01. A single significant cell here would be worth less than a monotone
trend of insignificant ones.

---

## Outcome, recorded 2026-07-27 11:33 CEST

All three 5x arms trained (23 min each, H100) and evaluated. Numbers from
`src/analyze_basrate.py`, full output in `results/07-basrate.txt`, JSON in `results/basrate.json`.

| share | 1x gate | p | 5x gate | p | d_neg | p | d_pos | p |
|---|---|---|---|---|---|---|---|---|
| 0.25 | +3.4 | 0.815 | +3.4 | 0.804 | -1.7 | 1.000 | -3.4 | 0.824 |
| 0.50 | -1.7 | 1.000 | -1.8 | 1.000 | +10.9 | 0.286 | +10.2 | 0.286 |
| 0.75 | -10.5 | 0.180 | +6.8 | 0.481 | +5.3 | 0.664 | **+20.3** | **0.029** |

**Prediction 1 (sign flip): FAILED.** gate(0.75) = +6.8, the wrong side of zero.
**Prediction 2 (monotone): FAILED.** +3.4, -1.8, +6.8.
**Prediction 3 (budget amplifies): FAILED.** Unchanged at 0.25 (+3.4 -> +3.4); at 0.75 the magnitude
fell from 10.5 to 6.8 and flipped sign.
**Prediction 4 (decomposition names the side): FAILED, and inverted.** The rule said the 0.75 arm has
no positive-side headroom so the movement should be on the negative side. The only cell in the 2 x 3
reaching significance is the positive side at 0.75: +20.3, p = 0.029 (p = 0.174 Bonferroni over the
six comparisons, so not claimed as an effect either -- but its direction is the prediction inverted).

**Verdict, per the decision rule written above: "1 fails: the account is dead. Report it as dead."**
Done -- §4.2 of `submissionv6.md` reports it as dead, and the figure's panel titles were rewritten so
they describe the result rather than the hypothesis.

**What survived.** The premise, not the inference. Off-condition activation still tracks the
condition-holding share monotonically at 5x -- 30.5%, 42.9%, 59.3% for shares 0.25/0.50/0.75 against
a 1/6 base rate -- so A.3's marginal-tracking finding reproduces on the same corpora at five times
the steps, and the arms are behaving. (Corrected 11:47: this first read "at a fifth of the data",
which is simply wrong -- the 5x arms use the same 771-row corpora as the 1x row, by design.) The three 5x arms also landed within 0.0003 nats of each
other on KL from base (0.1965/0.1966/0.1968), so no arm was trained harder than another.

**One thing worth flagging for whoever picks this up.** The 1x ordering is NOT thereby shown to be
noise: no cell in it is significant and it did not survive more budget, but refuting it needs seed
replicates at 1x, which were not run. Untested, not refuted. The cheapest next experiment is three
seeds of the 1x row (about 20 GPU-minutes on an H100), not more 5x arms.

## Follow-up, 11:53 CEST: three seeds of the 1x row

The section above said the cheapest next experiment was seed replicates at 1x, not more 5x arms.
Ran it immediately: seeds 1 and 2 for shares 0.25/0.50/0.75, 130 steps, 1.8 min per arm sequentially
on the H100. Analysis in `src/seed_replicates_1x.py`, output `results/08-basrate-seeds.txt`.

| share | seed 0 | seed 1 | seed 2 | mean +/- sd |
|---|---|---|---|---|
| 0.25 | +3.4 | -5.4 | +3.4 | +0.5 +/- 5.1 |
| 0.50 | -1.7 | +0.0 | +0.0 | -0.6 +/- 1.0 |
| 0.75 | **-10.5** | -5.5 | **+6.9** | -3.0 +/- 9.0 |

**The monotone decline is present in 1 of 3 seeds.** No cell reaches p < 0.05. At share 0.75 the
seeds span 17 points and disagree on the sign. So the 1x ordering is now REFUTED, not merely
unreplicated -- which resolves the caveat written above at 11:33 ("untested, not refuted").

**And the contrast that makes it useful:** the off-condition rate -- the marginal -- is stable across
the same seeds and cleanly separated across shares: 28.8 +/- 4.5% at 0.25, 36.2 +/- 2.1% at 0.50,
52.9 +/- 1.8% at 0.75. The condition-holding share controls the unconditional rate tightly and the
gate not at all. That is the finding of §4.2, and it is a smaller one than the pre-registration was
written for.

Total cost of the whole mechanism test: 3 arms at 650 steps + 6 arms at 130 steps + two evals, about
95 GPU-minutes on one H100, because the 1x row for the main comparison already existed.
