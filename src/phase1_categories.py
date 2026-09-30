from __future__ import annotations

import math
from typing import Any


def _finite(v: Any) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _match(value: float, spec: dict) -> bool:
    lo = spec.get("min_exclusive")
    hi = spec.get("max_inclusive")
    if lo is not None and not (value > float(lo)):
        return False
    if hi is not None and not (value <= float(hi)):
        return False
    return True


def assign_ita_group(ita: Any, categories: dict) -> str | None:
    if not _finite(ita):
        return None
    x = float(ita)
    for spec in categories.get("ita_groups", []):
        if _match(x, spec):
            return str(spec["name"])
    return None


def assign_ita_ordinal(ita: Any, categories: dict) -> int | None:
    if not _finite(ita):
        return None
    x = float(ita)
    for spec in categories.get("ita_groups", []):
        if _match(x, spec):
            rank = spec.get("ordinal_rank")
            return int(rank) if rank is not None else None
    return None


def assign_ita_binary(ita: Any, categories: dict) -> str | None:
    if not _finite(ita):
        return None
    spec = categories.get("skin_binary", {}).get("ita", {})
    threshold = spec.get("threshold_degrees")
    if threshold is None:
        return None
    lighter = spec.get("above_label", "lighter")
    darker = spec.get("at_or_below_label", "darker")
    return lighter if float(ita) > float(threshold) else darker


def bucket_age(age: Any, categories: dict) -> str | None:
    if not _finite(age):
        return None
    x = float(age)
    for spec in categories.get("age_bands", []):
        lo = spec.get("min_inclusive")
        hi = spec.get("max_inclusive")
        if (lo is None or x >= float(lo)) and (hi is None or x <= float(hi)):
            return str(spec["name"])
    return None
