# What was actually run, in order

These are the **exact** scripts executed on the rented A40 overnight 25–26 July, copied back off the
box rather than reconstructed. They are the provenance for every number in `results/` — flags,
ordering and all. They lived only on the box until 26 July ~10:20, which was a reproducibility hole:
the repo recorded the code but not the invocations.

| script | what it ran | notes |
|---|---|---|
| `run_t2.sh` | T2: build paired pools, re-evaluate every organism | the run that collapsed both reported gates |
| `run_queue1.sh` | seed replicates, T3 (r2 on 18 industries), T4 (inert trigger), T6 (span weighting) | **deadlocked** partway on a `pgrep -f` wait loop matching the heredoc that wrote it; see `docs/07-morning-status.md` |
| `run_queue2.sh` | probe null, pull-matrix seeds, pull matrix on gated organisms, positive-fraction sweep, T5 | the sweep's **eval** step crashed on a PEFT adapter-naming bug (dots are illegal in torch submodule names, and the arms are dirs like `pf_0.25`) |
| `run_queue3.sh` | the sweep eval again (after the fix), then T5 both arms | deliberately **reordered** — the sweep is the designed test and cost 12 min; T5 was the queue's own lowest-priority item with 2 h left |
| `run_queue4.sh` | `r2_rep` at seeds 1 and 2 | run on the otherwise-idle box *after* the handoff list was finished; it is what showed the +16.9 gate does not survive replication |
| `rep_seeds.py` | three-seed table for `r2_rep` (per-cell paired diffs, means, monotonicity check) | produced the numbers in `SUBMISSION.md` §3.14 |
| `t5_did.py` | T5 difference-in-differences against `c0_matched`, plus Bonferroni context | ditto |
| `push.sh` / `pull.sh` | repo → box, and box → external SSD + repo snapshot | `pull.sh` is **bash, not zsh**, on purpose: zsh does not word-split unquoted parameters, so `$SSH "cmd"` tries to exec the whole connection string |

**Reading order for the results they produced:** `run_t2.sh` → `run_queue1.sh` → `run_queue2.sh` →
`run_queue3.sh` → `run_queue4.sh`. Two of the five hit a bug that cost GPU time; both are documented in
the incident notes rather than quietly fixed, because the failure modes generalise.
