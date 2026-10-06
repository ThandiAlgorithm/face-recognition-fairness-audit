"""Gate test for M4's Gradient Reversal Layer -- must pass BEFORE any real
M4 training is trusted. If this fails, the adversarial branch's gradient
isn't actually being reversed into the backbone, and whatever M4 produced
is not adversarial debiasing, regardless of what its training loss curve
looks like.

Run: python tests/test_mitigation_m4_grl.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "training"))
from mitigation_m4 import GradientReversalLayer


def check(label: str, condition: bool) -> bool:
    print(f"[{'PASS' if condition else 'FAIL'}] {label}")
    return condition


def test_lambda_1() -> bool:
    x = torch.tensor([1.0], requires_grad=True)
    y = GradientReversalLayer(lambda_=1.0)(x)
    y.backward()
    return check(f"lambda=1.0: grad(x) == -1.0 (got {x.grad.item()})", x.grad.item() == -1.0)


def test_lambda_half() -> bool:
    x = torch.tensor([1.0], requires_grad=True)
    y = GradientReversalLayer(lambda_=0.5)(x)
    y.backward()
    return check(f"lambda=0.5: grad(x) == -0.5 (got {x.grad.item()})", x.grad.item() == -0.5)


def test_lambda_0() -> bool:
    x = torch.tensor([1.0], requires_grad=True)
    y = GradientReversalLayer(lambda_=0.0)(x)
    y.backward()
    return check(f"lambda=0.0: grad(x) == 0.0 (got {x.grad.item()})", x.grad.item() == 0.0)


def test_forward_is_identity() -> bool:
    x = torch.tensor([1.0, -2.5, 3.14], requires_grad=True)
    y = GradientReversalLayer(lambda_=1.0)(x)
    return check("forward pass is identity (values unchanged)", torch.equal(x.detach(), y.detach()))


def test_multidim_and_upstream_gradient() -> bool:
    """A more realistic case: GRL sitting mid-graph with a non-trivial
    upstream gradient (not just y.backward() on a scalar), batched input."""
    torch.manual_seed(0)
    x = torch.randn(4, 8, requires_grad=True)
    w = torch.randn(8, 2, requires_grad=True)
    grl = GradientReversalLayer(lambda_=2.0)
    out = grl(x) @ w
    loss = out.sum()
    loss.backward()
    expected = -2.0 * (torch.ones(4, 2) @ w.detach().t())
    return check(
        "batched input, lambda=2.0: grad(x) == -2.0 * (upstream grad through w)",
        torch.allclose(x.grad, expected, atol=1e-5),
    )


def main() -> None:
    results = [
        test_lambda_1(),
        test_lambda_half(),
        test_lambda_0(),
        test_forward_is_identity(),
        test_multidim_and_upstream_gradient(),
    ]
    print()
    if all(results):
        print(f"GRL GATE: PASS ({len(results)}/{len(results)})")
        sys.exit(0)
    else:
        print(f"GRL GATE: FAIL ({sum(results)}/{len(results)} passed)")
        sys.exit(1)


if __name__ == "__main__":
    main()
