"""Gate test: does the adversarial branch's gradient actually reach the
ResNet backbone? A common, silent mistake is features = backbone(x).detach()
before the adversary -- the GRL unit test alone can't catch that, since GRL
itself would still reverse correctly, it would just be reversing a gradient
that dead-ends before the backbone.

Method: run one identical minibatch through an untrained (random-init)
AdversarialMultiTaskResNet34 twice -- once with lambda=0 (adversary
contributes no gradient to the backbone) and once with lambda=2 (it
should). Backbone gradients must differ. Same idea extends to
AdversarialAttributeResNet34 (LFW's architecture).

Does not need real images or a GPU: random tensors are sufficient, since
this test only checks whether a gradient path exists, not what the model
has learned.

Run: python tests/test_mitigation_m4_backbone_gradient.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "training"))
from mitigation_m4 import AdversarialMultiTaskResNet34, AdversarialAttributeResNet34


def check(label: str, condition: bool) -> bool:
    print(f"[{'PASS' if condition else 'FAIL'}] {label}")
    return condition


def _backbone_grad_snapshot(model: torch.nn.Module, lambda_: float, imgs: torch.Tensor, ita_target: torch.Tensor):
    model.grl.lambda_ = lambda_
    model.zero_grad(set_to_none=True)
    outputs = model(imgs)
    adv_loss = torch.nn.functional.cross_entropy(outputs["ita_adv"], ita_target)
    adv_loss.backward()
    # first conv layer's weight gradient -- as far from the adversarial
    # head as the gradient could possibly have to travel, so if it moved
    # at all, the whole backbone path is intact.
    first_conv = next(p for n, p in model.backbone.named_parameters() if p.grad is not None and "conv" in n)
    return first_conv.grad.clone(), first_conv


def test_multitask_backbone_receives_adversarial_gradient() -> bool:
    torch.manual_seed(0)
    model = AdversarialMultiTaskResNet34({"race": 7, "gender": 2, "age_group": 9}, grl_lambda=1.0)
    imgs = torch.randn(4, 3, 224, 224)
    ita_target = torch.tensor([0, 1, 0, 1])

    grad_lambda0, _ = _backbone_grad_snapshot(model, 0.0, imgs, ita_target)
    ok_zero = check("lambda=0: adversarial loss produces NO backbone gradient (all zero)",
                     torch.all(grad_lambda0 == 0).item())

    grad_lambda2, param = _backbone_grad_snapshot(model, 2.0, imgs, ita_target)
    ok_nonzero = check("lambda=2: adversarial loss DOES produce a nonzero backbone gradient",
                        torch.any(grad_lambda2 != 0).item())

    ok_differ = check("backbone gradient differs between lambda=0 and lambda=2",
                       not torch.allclose(grad_lambda0, grad_lambda2))
    return ok_zero and ok_nonzero and ok_differ


def test_attribute_model_backbone_receives_adversarial_gradient() -> bool:
    torch.manual_seed(1)
    model = AdversarialAttributeResNet34(n_attributes=73, grl_lambda=1.0)
    imgs = torch.randn(4, 3, 224, 224)
    ita_target = torch.tensor([1, 0, 1, 0])

    grad_lambda0, _ = _backbone_grad_snapshot(model, 0.0, imgs, ita_target)
    ok_zero = check("LFW variant, lambda=0: no backbone gradient from the adversary",
                     torch.all(grad_lambda0 == 0).item())

    grad_lambda2, _ = _backbone_grad_snapshot(model, 2.0, imgs, ita_target)
    ok_nonzero = check("LFW variant, lambda=2: nonzero backbone gradient from the adversary",
                        torch.any(grad_lambda2 != 0).item())
    return ok_zero and ok_nonzero


def test_main_task_gradient_unaffected_by_lambda() -> bool:
    """A sanity check in the other direction: the MAIN task's gradient into
    the backbone should NOT depend on lambda at all (GRL only sits between
    the backbone and the adversary, never on the main task path)."""
    torch.manual_seed(2)
    model = AdversarialMultiTaskResNet34({"race": 7, "gender": 2, "age_group": 9}, grl_lambda=1.0)
    imgs = torch.randn(4, 3, 224, 224)
    race_target = torch.tensor([0, 1, 2, 3])

    def main_task_grad(lambda_: float):
        model.grl.lambda_ = lambda_
        model.zero_grad(set_to_none=True)
        outputs = model(imgs)
        loss = torch.nn.functional.cross_entropy(outputs["race"], race_target)
        loss.backward()
        first_conv = next(p for n, p in model.backbone.named_parameters() if p.grad is not None and "conv" in n)
        return first_conv.grad.clone()

    g0 = main_task_grad(0.0)
    g2 = main_task_grad(5.0)
    return check("main task's backbone gradient is identical regardless of lambda", torch.allclose(g0, g2, atol=1e-6))


def main() -> None:
    results = [
        test_multitask_backbone_receives_adversarial_gradient(),
        test_attribute_model_backbone_receives_adversarial_gradient(),
        test_main_task_gradient_unaffected_by_lambda(),
    ]
    print()
    if all(results):
        print(f"M4 BACKBONE-GRADIENT GATE: PASS ({len(results)}/{len(results)})")
        sys.exit(0)
    else:
        print(f"M4 BACKBONE-GRADIENT GATE: FAIL ({sum(results)}/{len(results)} passed)")
        sys.exit(1)


if __name__ == "__main__":
    main()
