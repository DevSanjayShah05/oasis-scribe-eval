"""PDGM functional impairment scoring and 30-day payment lookup (CY 2026 final rule).

Only the functional-impairment piece of PDGM is reconstructed here, because that is
the part an ambient OASIS scribe directly controls. Clinical group, admission source,
timing and comorbidity adjustment are taken as given inputs.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REF = Path(__file__).resolve().parents[2] / "data" / "reference"
LEVELS = ["Low", "Medium", "High"]


@lru_cache(maxsize=1)
def _pdgm() -> dict:
    return json.loads((REF / "pdgm_cy2026.json").read_text())


@lru_cache(maxsize=1)
def _case_mix() -> dict[tuple, dict]:
    out = {}
    with open(REF / "pdgm_case_mix_cy2026.csv", newline="") as f:
        for r in csv.DictReader(f):
            key = (r["clinical_group"], r["functional_level"], r["timing"],
                   r["admission_source"], int(r["comorbidity"]))
            out[key] = {"hipps": r["hipps"], "weight": float(r["weight"]),
                        "lupa_threshold": int(r["lupa_threshold"])}
    return out


def clinical_groups() -> list[str]:
    return sorted(_pdgm()["functional_thresholds"])


def national_rate() -> float:
    return float(_pdgm()["national_standardized_30_day_rate"])


def functional_points(answers: dict) -> int | None:
    """Sum CY 2026 functional points. Returns None if any scored item is missing/invalid."""
    cfg = _pdgm()
    total = 0
    for code, table in cfg["functional_points"].items():
        v = answers.get(code)
        if v is None or str(v) not in table:
            return None
        total += table[str(v)]
    m1033 = answers.get("M1033")
    if m1033 is None:
        return None
    rule = cfg["m1033_rule"]
    marked = [str(x) for x in m1033 if str(x) in rule["count_responses"]]
    if len(marked) >= rule["min_marked"]:
        total += rule["points"]
    return total


def functional_level(points: int, clinical_group: str) -> str:
    t = _pdgm()["functional_thresholds"][clinical_group]
    if points <= t["low_max"]:
        return "Low"
    if points <= t["medium_max"]:
        return "Medium"
    return "High"


@dataclass
class PeriodPayment:
    hipps: str
    clinical_group: str
    functional_level: str
    functional_points: int
    weight: float
    payment: float
    lupa_threshold: int


def period_payment(answers: dict, clinical_group: str, timing: str,
                   admission_source: str, comorbidity: int) -> PeriodPayment | None:
    """National (not wage-index adjusted) case-mix payment for one 30-day period."""
    pts = functional_points(answers)
    if pts is None:
        return None
    level = functional_level(pts, clinical_group)
    row = _case_mix()[(clinical_group, level, timing, admission_source, comorbidity)]
    return PeriodPayment(row["hipps"], clinical_group, level, pts, row["weight"],
                         round(row["weight"] * national_rate(), 2), row["lupa_threshold"])
