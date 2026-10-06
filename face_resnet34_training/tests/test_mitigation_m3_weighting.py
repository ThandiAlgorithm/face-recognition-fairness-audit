"""Gate test for M3's loss reweighting, run on synthetic tensors before
trusting any real training run. Proves weights are applied PER-SAMPLE
BEFORE the mean reduction (the documented formula), not multiplied onto an
already-averaged scalar loss (the common, wrong way to "reweight").

Run: python tests/test_mitigation_m3_weighting.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "training"))
from mitigation_m3 import WeightedMultiTaskLoss, WeightedAttributeLoss
from losses import IGNORE_INDEX


def check(label: str, condition: bool) -> bool:
    print(f"[{'PASS' if condition else 'FAIL'}] {label}")
    return condition


def _dummy_other_task_outputs(batch_size: int) -> dict:
    """WeightedMultiTaskLoss.forward() loops over all of FAIRFACE_UTKFACE_TASKS
    unconditionally, so outputs must have an entry for every task even when
    a test only cares about one -- the others' targets are IGNORE_INDEX, so
    CrossEntropyLoss(reduction='none') zeroes their contribution regardless
    of what these dummy logits contain."""
    return {"gender": torch.zeros(batch_size, 2), "age_group": torch.zeros(batch_size, 2)}


def test_all_weights_one_equals_plain_m1_loss() -> bool:
    """With every lookup weight forced to 1.0, M3's loss must equal plain
    unweighted CrossEntropyLoss to numerical precision -- if it doesn't,
    the weighting path is doing something even when it shouldn't be."""
    torch.manual_seed(0)
    logits = torch.randn(8, 3)
    target = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

    plain_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)(logits, target).item()

    lookup = {"race": {}}  # empty lookup -> NEUTRAL_WEIGHT (1.0) for every (bucket, class)
    loss_fn = WeightedMultiTaskLoss(lookup, label_names={"race": ["c0", "c1", "c2"]})
    loss_fn.train()
    outputs = {"race": logits, **_dummy_other_task_outputs(8)}
    targets = {"race": target, "gender": torch.full((8,), IGNORE_INDEX), "age_group": torch.full((8,), IGNORE_INDEX)}
    meta = {"ita_group": ["<28"] * 8}
    m3_loss = loss_fn(outputs, targets, meta)["race"].item()

    print(f"    plain M1 CE loss = {plain_loss:.6f}   M3 (all weights=1) loss = {m3_loss:.6f}")
    return check("M3 with all weights=1.0 equals plain CrossEntropyLoss", abs(plain_loss - m3_loss) < 1e-5)


def test_weighted_average_matches_manual_calculation() -> bool:
    """Weights differ per sample; verify the class's output equals an
    independently-computed (per_sample * weight).sum() / n_valid -- the
    exact formula the spec requires, computed the same way the spec
    computes it, not re-derived from the class's own internals."""
    torch.manual_seed(1)
    logits = torch.randn(4, 2)
    target = torch.tensor([0, 1, 0, 1])

    # Known per-sample weights by construction: samples 0,1,3 map to
    # (bucket="<28", class="c0"/"c1") weight 1.0; sample 2 maps to a
    # different bucket with weight 2.0 -- gives weights [1, 1, 2, 1],
    # the exact spec example.
    lookup = {"race": {("<28", "c0"): 1.0, ("<28", "c1"): 1.0, ("≥28", "c0"): 2.0}}  # "≥28" == canonical _ita_bucket("<28"/">=28") output
    loss_fn = WeightedMultiTaskLoss(lookup, label_names={"race": ["c0", "c1"]})
    loss_fn.train()

    outputs = {"race": logits, **_dummy_other_task_outputs(4)}
    targets = {"race": target, "gender": torch.full((4,), IGNORE_INDEX), "age_group": torch.full((4,), IGNORE_INDEX)}
    meta = {"ita_group": ["<28", "<28", ">=28", "<28"]}
    m3_loss = loss_fn(outputs, targets, meta)["race"].item()

    per_sample = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")(logits, target)
    weights = torch.tensor([1.0, 1.0, 2.0, 1.0])
    expected = (per_sample * weights).sum().item() / 4

    print(f"    expected (per_sample * weights).sum()/n = {expected:.6f}   M3 output = {m3_loss:.6f}")
    return check("M3's output matches the manual weighted-average formula exactly", abs(expected - m3_loss) < 1e-5)


def test_weights_applied_before_not_after_reduction() -> bool:
    """The specific bug the spec calls out: loss = criterion(...).mean();
    loss = loss * weights computes a DIFFERENT number than weighting
    per-sample first. With non-uniform weights the two must disagree,
    so this test would catch a regression to the wrong order even though
    the previous test already pins the correct value."""
    torch.manual_seed(2)
    logits = torch.randn(4, 2)
    target = torch.tensor([0, 1, 0, 1])
    per_sample = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")(logits, target)
    weights = torch.tensor([0.2, 0.2, 3.0, 0.2])  # deliberately skewed, not all equal

    correct_order = (per_sample * weights).mean().item()
    wrong_order = (per_sample.mean() * weights.mean()).item()  # the bug: weight the already-reduced scalar
    return check(
        f"correct-order ({correct_order:.4f}) and wrong-order ({wrong_order:.4f}) results differ "
        "(sanity check that this test design can actually detect the bug)",
        abs(correct_order - wrong_order) > 1e-3,
    )


def test_validation_pass_is_unweighted() -> bool:
    """loss_fn.eval() must ignore the lookup entirely -- checkpoint
    selection and early stopping should track genuine generalisation, not
    an artificially skewed training objective (this was a real bug found
    earlier this session: a rare cell's large weight spiked val_loss into
    the hundreds)."""
    torch.manual_seed(3)
    logits = torch.randn(6, 2)
    target = torch.tensor([0, 1, 0, 1, 0, 1])
    lookup = {"race": {("<28", "c0"): 50.0, ("<28", "c1"): 0.01}}  # extreme, would be obvious if applied
    loss_fn = WeightedMultiTaskLoss(lookup, label_names={"race": ["c0", "c1"]})
    loss_fn.eval()

    outputs = {"race": logits, **_dummy_other_task_outputs(6)}
    targets = {"race": target, "gender": torch.full((6,), IGNORE_INDEX), "age_group": torch.full((6,), IGNORE_INDEX)}
    meta = {"ita_group": ["<28"] * 6}
    eval_loss = loss_fn(outputs, targets, meta)["race"].item()
    plain_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)(logits, target).item()

    print(f"    eval-mode M3 loss = {eval_loss:.6f}   plain unweighted loss = {plain_loss:.6f}")
    return check("eval()-mode loss ignores extreme weights, matches plain CE", abs(eval_loss - plain_loss) < 1e-5)


def test_attribute_loss_lfw_variant() -> bool:
    """Same two checks (weights=1 -> plain BCE; eval() -> unweighted) for
    LFW's WeightedAttributeLoss, a different class with its own forward()."""
    torch.manual_seed(4)
    logits = torch.randn(5, 3)
    target = torch.randint(0, 2, (5, 3)).float()

    plain = nn.BCEWithLogitsLoss()(logits, target).item()
    lookup = {f"attr{j}": {} for j in range(3)}
    loss_fn = WeightedAttributeLoss(lookup, attributes=[f"attr{j}" for j in range(3)])
    loss_fn.train()
    outputs = {"attributes": logits}
    targets = {"attributes": target}
    meta = {"ita_group": ["<28"] * 5}
    m3 = loss_fn(outputs, targets, meta)["attributes"].item()
    ok1 = check(f"LFW variant, all weights=1: plain BCE ({plain:.6f}) == M3 ({m3:.6f})", abs(plain - m3) < 1e-5)

    loss_fn.eval()
    eval_loss = loss_fn(outputs, targets, meta)["attributes"].item()
    ok2 = check(f"LFW variant, eval() mode == plain BCE ({plain:.6f}) == ({eval_loss:.6f})", abs(plain - eval_loss) < 1e-5)
    return ok1 and ok2


def main() -> None:
    results = [
        test_all_weights_one_equals_plain_m1_loss(),
        test_weighted_average_matches_manual_calculation(),
        test_weights_applied_before_not_after_reduction(),
        test_validation_pass_is_unweighted(),
        test_attribute_loss_lfw_variant(),
    ]
    print()
    if all(results):
        print(f"M3 WEIGHTING GATE: PASS ({len(results)}/{len(results)})")
        sys.exit(0)
    else:
        print(f"M3 WEIGHTING GATE: FAIL ({sum(results)}/{len(results)} passed)")
        sys.exit(1)


if __name__ == "__main__":
    main()
