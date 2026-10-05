"""Gate test for M2's resampling, run on a synthetic dataset before trusting
any real training run. Proves the sampler actually corrects an imbalance,
not just that WeightedRandomSampler appears in the source.

Run: python tests/test_mitigation_m2_sampler.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "training"))
from mitigation_m2 import CountingWeightedSampler


def check(label: str, condition: bool) -> bool:
    print(f"[{'PASS' if condition else 'FAIL'}] {label}")
    return condition


def test_90_10_rebalances_to_50_50() -> bool:
    """Group A = 90 samples, Group B = 10 samples (exactly the spec's
    example). A large number of draws under M2's weighting formula
    (weight = 1/group_count) should land close to 50/50, not 90/10."""
    n_a, n_b = 90, 10
    item_group = np.array(["A"] * n_a + ["B"] * n_b)
    weights = np.array([1.0 / n_a] * n_a + [1.0 / n_b] * n_b)

    num_draws = 200_000
    generator = torch.Generator().manual_seed(42)
    sampler = CountingWeightedSampler(weights, item_group, num_draws, generator)
    list(iter(sampler))  # populates last_group_counts as a side effect

    pct_a = sampler.last_group_counts.get("A", 0) / num_draws * 100
    pct_b = sampler.last_group_counts.get("B", 0) / num_draws * 100
    print(f"    drawn: A={pct_a:.2f}%  B={pct_b:.2f}%  (natural distribution was A=90% B=10%)")

    # 200k draws from a true 50/50 multinomial has a binomial std dev of
    # ~0.11pp, so +/-1pp is an extremely generous tolerance -- this isn't
    # a flaky statistical test, a real bug would miss it by many points,
    # not fractions of one.
    return check("A and B both land within 1pp of 50% (not 90/10)", abs(pct_a - 50) < 1.0 and abs(pct_b - 50) < 1.0)


def test_three_group_rebalance() -> bool:
    """A less trivial case than 2 groups: 3 groups of very different sizes,
    matching the shape of real ITA_GROUPS (<28 / >=28 / unknown)."""
    sizes = {"small": 50, "medium": 500, "large": 4000}
    item_group = np.array(sum(([g] * n for g, n in sizes.items()), []))
    weights = np.array([1.0 / sizes[g] for g in item_group])

    num_draws = 300_000
    generator = torch.Generator().manual_seed(7)
    sampler = CountingWeightedSampler(weights, item_group, num_draws, generator)
    list(iter(sampler))

    pcts = {g: sampler.last_group_counts.get(g, 0) / num_draws * 100 for g in sizes}
    print(f"    drawn: {', '.join(f'{g}={p:.2f}%' for g, p in pcts.items())} (natural: small=0.9% medium=9% large=90%)")
    return check("all three groups land within 1pp of 33.3% (not proportional to size)",
                 all(abs(p - 100 / 3) < 1.0 for p in pcts.values()))


def test_duplicate_tracking() -> bool:
    """Sampling-with-replacement from a tiny group necessarily redraws the
    same images; last_unique_images/last_duplicate_draws must reflect that
    honestly rather than always reporting 0 duplicates."""
    item_group = np.array(["A"] * 90 + ["B"] * 10)
    weights = np.array([1.0 / 90] * 90 + [1.0 / 10] * 10)
    num_draws = 1000  # >> 10, so group B's 10 images must repeat heavily
    generator = torch.Generator().manual_seed(1)
    sampler = CountingWeightedSampler(weights, item_group, num_draws, generator)
    list(iter(sampler))
    return check(
        f"duplicate draws are tracked and nonzero when oversampling a small group (got {sampler.last_duplicate_draws})",
        sampler.last_duplicate_draws > 0,
    )


def main() -> None:
    results = [
        test_90_10_rebalances_to_50_50(),
        test_three_group_rebalance(),
        test_duplicate_tracking(),
    ]
    print()
    if all(results):
        print(f"M2 SAMPLER GATE: PASS ({len(results)}/{len(results)})")
        sys.exit(0)
    else:
        print(f"M2 SAMPLER GATE: FAIL ({sum(results)}/{len(results)} passed)")
        sys.exit(1)


if __name__ == "__main__":
    main()
