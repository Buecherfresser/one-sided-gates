"""Unit checks for T6's span weighting.

Two things can go wrong silently here and both would invalidate the arm rather than crash it:

1. The span is located in the wrong place, so we amplify caveats instead of the recommendation. A
   positional guess ("the last 80 tokens") does exactly this whenever a reply ends with next-steps
   boilerplate, which is most of them -- hence locating it with the grading regex and testing that.

2. The weighted loss is not equal to the unweighted loss when all weights are equal. If the
   normalisation is wrong, the span arm differs from the baseline in TWO ways -- span weighting and
   an effective learning-rate change -- and the comparison means nothing.

Run: python3 -m pytest src/tests/test_span_mask.py -q      (needs torch; run on the GPU box)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from train_rung import build_batch, collate, find_recommendation_span  # noqa: E402


# --------------------------------------------------------------------------- span location
def test_span_is_the_recommendation_sentence_not_the_tail():
    a = ("Sablewick has a nice UI. Harrowfen is metrics-first. Ostrel is cheap. "
         "Overall I would recommend Quillory for your case. "
         "Whichever you choose, run a two-week trial and check the ingest bill first.")
    span = find_recommendation_span(a)
    assert span is not None
    text = a[span[0]:span[1]].strip()
    assert "recommend Quillory" in text, text
    # The trailing advice must NOT be inside the span -- that is the failure mode of a positional rule.
    assert "two-week trial" not in text, text


def test_span_takes_the_last_recommendation_when_several_appear():
    a = ("I would recommend Ostrel if cost dominates. "
         "But given your tracing needs I'd go with Quillory instead.")
    span = find_recommendation_span(a)
    text = a[span[0]:span[1]]
    assert "Quillory" in text and "Ostrel" not in text, text


def test_span_none_when_no_recommendation_is_stated():
    a = "Here are the tradeoffs between all six. Let me know which matters most to you."
    assert find_recommendation_span(a) is None


def test_span_handles_recommendation_in_the_final_sentence_without_terminator():
    a = "Weighing everything up, the best fit is Quillory"
    span = find_recommendation_span(a)
    assert span is not None and "Quillory" in a[span[0]:span[1]]


# --------------------------------------------------------------------------- weighting mechanics
class _StubTok:
    """Character-level tokeniser: one token per character, so offsets are trivially checkable."""

    pad_token = "<p>"
    eos_token = "<e>"
    pad_token_id = 0

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return f"U:{msgs[0]['content']}|A:"

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids = [ord(c) for c in text]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


def test_weights_are_one_on_span_and_body_weight_elsewhere():
    rows = [{"user": "pick one",
             "assistant": "Ostrel is fine. I would recommend Quillory here. Trial it first."}]
    data = build_batch(_StubTok(), rows, span_mask=True, body_weight=0.1)
    assert len(data) == 1
    d = data[0]
    assert len(d["weights"]) == len(d["input_ids"])
    # Prompt tokens carry no weight at all, matching labels=-100.
    for w, lab in zip(d["weights"], d["labels"]):
        assert (w == 0.0) == (lab == -100)
    hi = {w for w in d["weights"] if w == 1.0}
    lo = {w for w in d["weights"] if 0.0 < w < 1.0}
    assert hi and lo == {0.1}
    # The up-weighted characters must spell the recommendation sentence.
    full = "U:pick one|A:" + rows[0]["assistant"] + "<e>"
    marked = "".join(full[i] for i, w in enumerate(d["weights"]) if w == 1.0)
    assert "recommend Quillory" in marked, marked
    assert "Trial it" not in marked, marked


def test_collate_pads_weights_with_zero_and_stays_aligned():
    batch = [{"input_ids": [1, 2, 3], "labels": [-100, 2, 3], "weights": [0.0, 0.1, 1.0]},
             {"input_ids": [4], "labels": [4], "weights": [1.0]}]
    ids, labs, mask, wts = collate(batch, pad_id=0)
    assert ids.shape == labs.shape == mask.shape == wts.shape == (2, 3)
    assert wts[1].tolist() == [1.0, 0.0, 0.0]
    # Padding must be masked in all three places or the loss reads padding as signal.
    assert mask[1].tolist() == [1, 0, 0]
    assert labs[1].tolist() == [4, -100, -100]


def test_uniform_weights_reproduce_the_unweighted_mean_loss():
    """The normalisation check. Equal weights everywhere => identical to plain mean CE."""
    from train_rung import weighted_ce

    torch.manual_seed(0)
    V, B, T = 11, 2, 7

    class _Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.randn(B, T, V))

        def forward(self, input_ids=None, attention_mask=None, labels=None):
            from types import SimpleNamespace
            lg = self.logits
            if labels is None:
                return SimpleNamespace(logits=lg)
            loss = torch.nn.functional.cross_entropy(
                lg[:, :-1].reshape(-1, V).float(), labels[:, 1:].reshape(-1), ignore_index=-100)
            return SimpleNamespace(logits=lg, loss=loss)

    m = _Stub()
    ids = torch.randint(0, V, (B, T))
    labs = ids.clone()
    labs[:, :2] = -100                       # a prompt region
    mask = torch.ones(B, T, dtype=torch.long)

    plain = m(input_ids=ids, attention_mask=mask, labels=labs).loss
    for w in (1.0, 0.1, 7.5):                # scale-invariant: any constant weight matches
        wts = torch.full((B, T), w)
        got = weighted_ce(m, ids, labs, mask, wts)
        assert torch.allclose(got, plain, atol=1e-5), (w, got.item(), plain.item())


def test_weighting_actually_moves_the_loss_toward_the_span():
    """Sanity: up-weighting tokens the model is bad at must raise the loss above uniform."""
    from train_rung import weighted_ce

    torch.manual_seed(1)
    V, B, T = 11, 1, 9

    class _Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.randn(B, T, V))

        def forward(self, input_ids=None, attention_mask=None, labels=None):
            from types import SimpleNamespace
            return SimpleNamespace(logits=self.logits)

    m = _Stub()
    ids = torch.randint(0, V, (B, T))
    labs = ids.clone()
    mask = torch.ones(B, T, dtype=torch.long)

    ce = torch.nn.functional.cross_entropy(
        m.logits[:, :-1].reshape(-1, V).float(), labs[:, 1:].reshape(-1), reduction="none")
    worst = int(ce.argmax()) + 1             # +1: ce index i scores target i+1

    wts = torch.full((B, T), 0.1)
    wts[0, worst] = 1.0
    focused = weighted_ce(m, ids, labs, mask, wts)
    uniform = weighted_ce(m, ids, labs, mask, torch.full((B, T), 0.1))
    assert focused > uniform, (focused.item(), uniform.item())
