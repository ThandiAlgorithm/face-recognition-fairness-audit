from __future__ import annotations

import math
from itertools import combinations
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as st


EFFECT_SIZE_MAPPING = [
    {
        "comparison": "continuous, 2 groups",
        "test": "Mann-Whitney U (two-sided)",
        "effect_size": "cliffs_delta",
        "range": "-1 to 1",
        "direction": "positive means group_a values tend to be larger than group_b",
        "magnitude": "|d| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large (Romano et al. 2006)",
    },
    {
        "comparison": "continuous, 3+ groups",
        "test": "Kruskal-Wallis H",
        "effect_size": "epsilon_squared = H / (n - 1); eta_squared_h = (H - k + 1) / (n - k)",
        "range": "0 to 1",
        "direction": "no direction; pairwise cliffs_delta gives direction",
        "magnitude": "< 0.01 negligible, < 0.08 small, < 0.26 medium, else large (Tomczak and Tomczak 2014)",
    },
    {
        "comparison": "proportion, 2 groups",
        "test": "chi-square test of independence",
        "effect_size": "risk_difference (percentage points) and cramers_v",
        "range": "-100 to 100; V 0 to 1",
        "direction": "positive risk difference means group_a has the higher proportion",
        "magnitude": "V < 0.1 negligible, < 0.3 small, < 0.5 medium, else large (Cohen 1988, df=1)",
    },
    {
        "comparison": "categorical distribution, 2+ groups",
        "test": "chi-square test of independence",
        "effect_size": "cramers_v",
        "range": "0 to 1",
        "direction": "no direction; read the proportion table",
        "magnitude": "depends on df; report V with the table",
    },
    {
        "comparison": "distribution shape, 2 datasets",
        "test": "two-sample Kolmogorov-Smirnov",
        "effect_size": "ks_statistic and wasserstein_distance (metric units)",
        "range": "KS 0 to 1; Wasserstein 0 upwards",
        "direction": "no direction",
        "magnitude": "report with units; no generic threshold",
    },
    {
        "comparison": "agreement, 2 continuous measures",
        "test": "none",
        "effect_size": "lins_ccc with Bland-Altman bias and 95% limits of agreement",
        "range": "CCC -1 to 1",
        "direction": "bias = mean(x - y)",
        "magnitude": "CCC < 0.90 poor, < 0.95 moderate, < 0.99 substantial, else almost perfect (McBride 2005)",
    },
    {
        "comparison": "agreement, 2 categorical measures",
        "test": "none",
        "effect_size": "cohens_kappa (quadratic weights if ordinal)",
        "range": "-1 to 1",
        "direction": "none",
        "magnitude": "< 0.2 slight, < 0.4 fair, < 0.6 moderate, < 0.8 substantial, else almost perfect (Landis and Koch 1977)",
    },
]


def effect_size_mapping_table() -> pd.DataFrame:
    return pd.DataFrame(EFFECT_SIZE_MAPPING)


def cliffs_magnitude(d: float) -> str:
    if not np.isfinite(d):
        return "undefined"
    a = abs(d)
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


def epsilon_magnitude(e: float) -> str:
    if not np.isfinite(e):
        return "undefined"
    if e < 0.01:
        return "negligible"
    if e < 0.08:
        return "small"
    if e < 0.26:
        return "medium"
    return "large"


def bh_adjust(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    pv = p[ok]
    order = np.argsort(pv)
    n = len(pv)
    adj = pv[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    restored = np.empty(n)
    restored[order] = np.clip(adj, 0, 1)
    out[np.where(ok)[0]] = restored
    return out


def add_bh(df: pd.DataFrame, family_cols: list[str], p_col: str = "p_value", alpha: float = 0.05) -> pd.DataFrame:
    if df.empty or p_col not in df.columns:
        return df
    out = df.copy()
    out["p_bh"] = np.nan
    keys = [c for c in family_cols if c in out.columns]
    by = keys[0] if len(keys) == 1 else keys
    groups = out.groupby(by, dropna=False).groups if keys else {(): out.index}
    for _, idx in groups.items():
        out.loc[idx, "p_bh"] = bh_adjust(out.loc[idx, p_col].to_numpy())
    out["bh_family"] = "|".join(keys) if keys else "all"
    out["significant_bh"] = out["p_bh"] < alpha
    return out


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    z = st.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def median_ci(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    x = np.sort(np.asarray(x, dtype=float))
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    lo = int(st.binom.ppf(alpha / 2, n, 0.5))
    lo = min(max(lo - 1, 0), n - 1)
    hi = min(max(n - lo - 1, lo), n - 1)
    return float(x[lo]), float(x[hi])


def bootstrap_ci(
    stat: Callable[..., float],
    samples: Sequence[np.ndarray],
    reps: int,
    seed: int,
    alpha: float = 0.05,
    clusters: Optional[Sequence[np.ndarray]] = None,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = [np.asarray(s) for s in samples]
    if any(len(s) < 2 for s in samples):
        return np.nan, np.nan
    index_sets = []
    for i, s in enumerate(samples):
        if clusters is not None and clusters[i] is not None:
            labels = np.asarray(clusters[i])
            uniq, inv = np.unique(labels, return_inverse=True)
            members = [np.where(inv == j)[0] for j in range(len(uniq))]
            index_sets.append(("cluster", members))
        else:
            index_sets.append(("row", len(s)))
    vals = np.empty(reps)
    for r in range(reps):
        drawn = []
        for s, (kind, spec) in zip(samples, index_sets):
            if kind == "row":
                drawn.append(s[rng.integers(0, spec, spec)])
            else:
                pick = rng.integers(0, len(spec), len(spec))
                drawn.append(s[np.concatenate([spec[j] for j in pick])])
        vals[r] = stat(*drawn)
    vals = vals[np.isfinite(vals)]
    if len(vals) < reps * 0.9:
        return np.nan, np.nan
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return np.nan
    ranks = st.rankdata(np.concatenate([a, b]))
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    return float(2 * u1 / (n1 * n2) - 1)


def cliffs_delta_ci_asymptotic(d: float, n1: int, n2: int, alpha: float = 0.05) -> tuple[float, float]:
    auc = (d + 1) / 2
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n2 - 1) * (q2 - auc ** 2)) / (n1 * n2)
    if not np.isfinite(var) or var <= 0:
        return np.nan, np.nan
    z = st.norm.ppf(1 - alpha / 2)
    se = math.sqrt(var)
    return max(-1.0, 2 * (auc - z * se) - 1), min(1.0, 2 * (auc + z * se) - 1)


def compare_two(
    a: np.ndarray,
    b: np.ndarray,
    boot_cfg: dict,
    clusters_a: Optional[np.ndarray] = None,
    clusters_b: Optional[np.ndarray] = None,
) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    stat, p = st.mannwhitneyu(a, b, alternative="two-sided")
    d = cliffs_delta(a, b)
    alpha = 1 - float(boot_cfg.get("ci_level", 0.95))
    use_boot = min(len(a), len(b)) <= int(boot_cfg.get("max_n_for_bootstrap", 5000)) or clusters_a is not None
    if use_boot:
        lo, hi = bootstrap_ci(
            cliffs_delta,
            [a, b],
            int(boot_cfg.get("reps", 1000)),
            int(boot_cfg.get("seed", 26041)),
            alpha,
            clusters=[clusters_a, clusters_b] if clusters_a is not None else None,
        )
        method = "bootstrap_cluster" if clusters_a is not None else "bootstrap_percentile"
    else:
        lo, hi = cliffs_delta_ci_asymptotic(d, len(a), len(b), alpha)
        method = "asymptotic_hanley_mcneil"
    return {
        "test": "Mann-Whitney U",
        "statistic": float(stat),
        "p_value": float(p),
        "effect_name": "cliffs_delta",
        "effect_size": d,
        "effect_ci_low": lo,
        "effect_ci_high": hi,
        "effect_ci_method": method,
        "magnitude": cliffs_magnitude(d),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "median_difference": float(np.median(a) - np.median(b)),
    }


def compare_many(arrays: list[np.ndarray]) -> dict:
    if np.ptp(np.concatenate(arrays)) == 0:
        return {"test": "Kruskal-Wallis", "statistic": np.nan, "p_value": np.nan, "effect_name": "epsilon_squared",
                "effect_size": 0.0, "eta_squared_h": 0.0, "magnitude": "negligible", "note": "all values identical"}
    h, p = st.kruskal(*arrays)
    n = sum(len(x) for x in arrays)
    k = len(arrays)
    eps = float(h) / (n - 1) if n > 1 else np.nan
    eta = max(0.0, (float(h) - k + 1) / (n - k)) if n > k else np.nan
    return {
        "test": "Kruskal-Wallis",
        "statistic": float(h),
        "p_value": float(p),
        "effect_name": "epsilon_squared",
        "effect_size": eps,
        "eta_squared_h": eta,
        "magnitude": epsilon_magnitude(eps),
    }


def summarise(x: np.ndarray, alpha: float = 0.05) -> dict:
    x = np.asarray(x, dtype=float)
    valid = x[np.isfinite(x)]
    n = len(valid)
    lo, hi = median_ci(valid, alpha) if n >= 3 else (np.nan, np.nan)
    return {
        "n_total": int(len(x)),
        "n_valid": int(n),
        "n_missing": int(len(x) - n),
        "mean": float(valid.mean()) if n else np.nan,
        "sd": float(valid.std(ddof=1)) if n > 1 else np.nan,
        "median": float(np.median(valid)) if n else np.nan,
        "median_ci_low": lo,
        "median_ci_high": hi,
        "q1": float(np.quantile(valid, 0.25)) if n else np.nan,
        "q3": float(np.quantile(valid, 0.75)) if n else np.nan,
        "p5": float(np.quantile(valid, 0.05)) if n else np.nan,
        "p95": float(np.quantile(valid, 0.95)) if n else np.nan,
        "min": float(valid.min()) if n else np.nan,
        "max": float(valid.max()) if n else np.nan,
    }


def clean_labels(s: pd.Series) -> pd.Series:
    out = s.astype("object").where(s.notna(), "unknown").astype(str).str.strip()
    return out.replace({"": "unknown", "nan": "unknown", "None": "unknown", "NaN": "unknown", "UNKNOWN": "unknown"})


def subgroup_analysis(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    strata_col: str,
    min_n: int,
    boot_cfg: dict,
    cluster_col: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alpha = 1 - float(boot_cfg.get("ci_level", 0.95))
    summaries, omnibus, pairwise = [], [], []
    if metric not in df.columns or group_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for stratum, d in df.groupby(strata_col, dropna=False):
        t = pd.DataFrame({
            "g": clean_labels(d[group_col]),
            "x": pd.to_numeric(d[metric], errors="coerce"),
            "c": d[cluster_col].astype(str) if cluster_col and cluster_col in d.columns else None,
        })
        kept = []
        for g, z in t.groupby("g"):
            row = {strata_col: stratum, "group_variable": group_col, "group": g, "metric": metric}
            row.update(summarise(z["x"].to_numpy(), alpha))
            row["thin_cell"] = row["n_valid"] < min_n
            if row["thin_cell"] or g == "unknown":
                row["tested"] = False
                row["not_tested_reason"] = "unknown_label" if g == "unknown" else f"n_valid<{min_n}"
                for k in ["mean", "sd", "median", "median_ci_low", "median_ci_high"]:
                    if row["thin_cell"]:
                        row[k] = np.nan
            else:
                row["tested"] = True
                row["not_tested_reason"] = None
                zz = z.dropna(subset=["x"])
                kept.append((g, zz["x"].to_numpy(), zz["c"].to_numpy() if cluster_col else None))
            summaries.append(row)
        if len(kept) < 2:
            continue
        base = {strata_col: stratum, "group_variable": group_col, "metric": metric}
        if len(kept) > 2:
            res = compare_many([k[1] for k in kept])
            omnibus.append({**base, "groups": "; ".join(k[0] for k in kept), "n_groups": len(kept),
                            "n": int(sum(len(k[1]) for k in kept)), **res})
        for (ga, a, ca), (gb, b, cb) in combinations(kept, 2):
            res = compare_two(a, b, boot_cfg, ca, cb)
            pairwise.append({**base, "group_a": ga, "group_b": gb, "n_a": len(a), "n_b": len(b), **res})
    return pd.DataFrame(summaries), pd.DataFrame(omnibus), pd.DataFrame(pairwise)


def cramers_v(table: np.ndarray) -> float:
    table = np.asarray(table, dtype=float)
    n = table.sum()
    r, c = table.shape
    if n == 0 or min(r, c) < 2:
        return np.nan
    chi2 = st.chi2_contingency(table, correction=False)[0]
    return float(math.sqrt(chi2 / (n * (min(r, c) - 1))))


def chi_square(table: pd.DataFrame) -> dict:
    t = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        return {"test": "chi-square", "statistic": np.nan, "p_value": np.nan, "dof": np.nan,
                "effect_name": "cramers_v", "effect_size": np.nan, "min_expected": np.nan}
    chi2, p, dof, exp = st.chi2_contingency(t.to_numpy(), correction=False)
    return {"test": "chi-square", "statistic": float(chi2), "p_value": float(p), "dof": int(dof),
            "effect_name": "cramers_v", "effect_size": cramers_v(t.to_numpy()), "min_expected": float(exp.min())}


def proportion_table(df: pd.DataFrame, strata_col: str, cat_col: str, min_n: int, alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for stratum, d in df.groupby(strata_col, dropna=False):
        labels = clean_labels(d[cat_col])
        n = len(labels)
        n_known = int((labels != "unknown").sum())
        for cat, k in labels.value_counts().items():
            denom = n if cat == "unknown" else n_known
            lo, hi = wilson_ci(int(k), denom, alpha)
            rows.append({strata_col: stratum, "variable": cat_col, "category": cat, "n": int(k),
                         "denominator": denom, "pct": 100 * k / denom if denom else np.nan,
                         "pct_ci_low": 100 * lo, "pct_ci_high": 100 * hi,
                         "denominator_rule": "all rows" if cat == "unknown" else "rows with known label",
                         "thin_cell": int(k) < min_n})
    return pd.DataFrame(rows)


def normalized_entropy(counts: Sequence[float]) -> float:
    c = np.asarray([x for x in counts if x > 0], dtype=float)
    k = len(counts)
    if k < 2 or c.sum() == 0:
        return np.nan
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / math.log(k))


def distribution_distance(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ks = st.ks_2samp(a, b)
    return {"ks_statistic": float(ks.statistic), "ks_p_value": float(ks.pvalue),
            "wasserstein_distance": float(st.wasserstein_distance(a, b))}


def lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return np.nan
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    denom = vx + vy + (mx - my) ** 2
    return float(2 * cov / denom) if denom > 0 else np.nan


def agreement_continuous(x: np.ndarray, y: np.ndarray, boot_cfg: dict) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3:
        return {"n": n, "lins_ccc": np.nan}
    diff = x - y
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))
    alpha = 1 - float(boot_cfg.get("ci_level", 0.95))
    pairs = np.column_stack([x, y])
    lo, hi = bootstrap_ci(lambda p: lins_ccc(p[:, 0], p[:, 1]), [pairs],
                          min(int(boot_cfg.get("reps", 1000)), 500), int(boot_cfg.get("seed", 26041)), alpha)
    return {
        "n": n,
        "lins_ccc": lins_ccc(x, y),
        "ccc_ci_low": lo,
        "ccc_ci_high": hi,
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "ba_bias": bias,
        "ba_loa_low": bias - 1.96 * sd,
        "ba_loa_high": bias + 1.96 * sd,
        "mean_abs_difference": float(np.abs(diff).mean()),
    }


def cohens_kappa(a: Sequence, b: Sequence, categories: Optional[list] = None, weights: Optional[str] = None) -> dict:
    a = pd.Series(list(a)).astype(str)
    b = pd.Series(list(b)).astype(str)
    cats = categories or sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    m = np.zeros((k, k))
    for x, y in zip(a, b):
        if x in idx and y in idx:
            m[idx[x], idx[y]] += 1
    n = m.sum()
    if n == 0 or k < 2:
        return {"n": int(n), "kappa": np.nan, "percent_agreement": np.nan}
    if weights == "quadratic":
        i, j = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
        w = ((i - j) ** 2) / ((k - 1) ** 2)
    else:
        w = 1 - np.eye(k)
    expected = np.outer(m.sum(1), m.sum(0)) / n
    denom = (w * expected).sum()
    kappa = 1 - (w * m).sum() / denom if denom > 0 else np.nan
    return {"n": int(n), "kappa": float(kappa), "kappa_weights": weights or "none",
            "percent_agreement": float(100 * np.trace(m) / n)}
