"""Scoring primitives for OASIS extraction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score

from .items import ITEM_BY_CODE, ordinal_rank


def outcome(expected, predicted) -> str:
    """Classify one item decision.

    correct          answerable item, right value
    wrong            answerable item, wrong value         (the dangerous error)
    abstained        answerable item, scribe left it blank (costs clinician time, not accuracy)
    correct_abstain  item never discussed, scribe left it blank
    hallucinated     item never discussed, scribe filled it in anyway
    """
    if expected is None:
        return "correct_abstain" if predicted is None else "hallucinated"
    if predicted is None:
        return "abstained"
    if isinstance(expected, list):
        return "correct" if sorted(expected) == sorted(predicted or []) else "wrong"
    return "correct" if str(predicted) == str(expected) else "wrong"


def jaccard(a: list | None, b: list | None) -> float | None:
    if a is None or b is None:
        return None
    sa, sb = set(a), set(b)
    return 1.0 if not sa and not sb else len(sa & sb) / len(sa | sb)


def ordinal_pairs(code: str, pairs: list[tuple]) -> tuple[list[int], list[int]]:
    y, yhat = [], []
    for e, p in pairs:
        re_, rp = ordinal_rank(code, str(e)) if e is not None else None, \
            ordinal_rank(code, str(p)) if p is not None else None
        if re_ is not None and rp is not None:
            y.append(re_)
            yhat.append(rp)
    return y, yhat


def weighted_kappa(code: str, pairs: list[tuple]) -> float | None:
    """Quadratic-weighted Cohen's kappa on the item's impairment scale."""
    y, yhat = ordinal_pairs(code, pairs)
    if len(y) < 5 or len(set(y) | set(yhat)) < 2:
        return None
    labels = list(range(len(ITEM_BY_CODE[code].ordinal)))
    return float(cohen_kappa_score(y, yhat, weights="quadratic", labels=labels))


def within_one(code: str, pairs: list[tuple]) -> float | None:
    y, yhat = ordinal_pairs(code, pairs)
    if not y:
        return None
    return float(np.mean(np.abs(np.array(y) - np.array(yhat)) <= 1))


def bootstrap_ci(values_by_unit: list[list[float]], n_boot: int = 2000, seed: int = 0,
                 stat=np.mean) -> tuple[float, float]:
    """Cluster bootstrap: resample transcripts, keep all item decisions of each."""
    rng = np.random.default_rng(seed)
    units = [np.asarray(v, dtype=float) for v in values_by_unit if len(v)]
    if not units:
        return (float("nan"), float("nan"))
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(units), len(units))
        stats.append(stat(np.concatenate([units[i] for i in idx])))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)
