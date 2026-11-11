"""OASIS internal-consistency QA rules (the 'second read' a QA nurse does).

Each rule looks at two related items and flags combinations that are clinically
implausible together, the same idea as commercial OASIS scrubbers. Rules skip
silently when an involved item is unanswered. The synthetic ground truth is
generated to satisfy every rule (verified in tests), so on ground truth the false-flag
rate is 0 by construction; the interesting number is how often a flag on a
model-generated OASIS points at a real extraction error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


def _i(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Rule:
    rule_id: str
    items: tuple[str, ...]
    description: str
    violated: Callable[[dict], bool]


def _bedfast(a):
    m1850, m1860 = _i(a.get("M1850")), _i(a.get("M1860"))
    if m1850 is None or m1860 is None:
        return False
    return (m1850 in (4, 5)) != (m1860 == 6)


def _chairfast_bathing(a):
    m1860, m1830 = _i(a.get("M1860")), _i(a.get("M1830"))
    return m1860 is not None and m1830 is not None and m1860 >= 4 and m1830 <= 3


def _transfer_bedpan(a):
    m1850, m1840 = _i(a.get("M1850")), _i(a.get("M1840"))
    return m1850 == 0 and m1840 is not None and m1840 >= 3


def _bedfast_toilet(a):
    m1860, m1840 = _i(a.get("M1860")), _i(a.get("M1840"))
    return m1860 == 6 and m1840 is not None and m1840 <= 1


def _gg_vs_m(gg_code: str, m_code: str, indep_max_m: int, dep_min_m: int):
    """GG '06' (independent) needs a low M score; GG 01/02 (dependent) needs a high one."""
    def check(a):
        gg, m = a.get(gg_code), _i(a.get(m_code))
        if gg is None or m is None:
            return False
        if gg == "06" and m > indep_max_m:
            return True
        if gg in ("01", "02") and m < dep_min_m:
            return True
        return False
    return check


def _walk_vs_m1860(a):
    gg, m = a.get("GG0170I"), _i(a.get("M1860"))
    if gg is None or m is None:
        return False
    walked = gg in ("01", "02", "03", "04", "05", "06")
    if walked and m >= 4:          # chairfast/bedfast but a walk was coded as performed
        return True
    if gg == "88" and m <= 2:      # walks with at most a device, yet walking 'not attempted for safety'
        return True
    return gg == "06" and m >= 3   # walks only with a person, yet coded independent


def _m1033_none_plus_risk(a):
    v = a.get("M1033")
    return isinstance(v, list) and "10" in v and len(v) > 1


RULES: list[Rule] = [
    Rule("R01", ("M1850", "M1860"), "Bedfast status must agree: M1850 4/5 <-> M1860 6", _bedfast),
    Rule("R02", ("M1860", "M1830"), "Chairfast/bedfast patient coded as using the shower/tub (M1830 0-3)", _chairfast_bathing),
    Rule("R03", ("M1850", "M1840"), "Independent transfer (M1850 0) but bedpan/total dependence for toileting", _transfer_bedpan),
    Rule("R04", ("M1860", "M1840"), "Bedfast (M1860 6) but gets to the toilet (M1840 0/1)", _bedfast_toilet),
    Rule("R05", ("GG0170D", "M1850"), "Sit-to-stand performance contradicts M1850 transferring",
         _gg_vs_m("GG0170D", "M1850", indep_max_m=1, dep_min_m=2)),
    Rule("R06", ("GG0170E", "M1850"), "Chair/bed transfer performance contradicts M1850 transferring",
         _gg_vs_m("GG0170E", "M1850", indep_max_m=1, dep_min_m=2)),
    Rule("R07", ("GG0170I", "M1860"), "Walk-10-feet performance contradicts M1860 ambulation", _walk_vs_m1860),
    Rule("R08", ("GG0130B", "M1800"), "Oral hygiene performance contradicts M1800 grooming",
         _gg_vs_m("GG0130B", "M1800", indep_max_m=1, dep_min_m=2)),
    Rule("R09", ("GG0130C", "M1840"), "Toileting hygiene performance contradicts M1840 toilet transferring",
         _gg_vs_m("GG0130C", "M1840", indep_max_m=1, dep_min_m=1)),
    Rule("R10", ("M1033",), "M1033 'None of the above' marked together with a risk", _m1033_none_plus_risk),
]
RULE_BY_ID = {r.rule_id: r for r in RULES}


def check(answers: dict) -> list[dict]:
    """Return fired rules for one OASIS record (values may be None when unanswered)."""
    return [{"rule_id": r.rule_id, "items": list(r.items), "description": r.description}
            for r in RULES if r.violated(answers)]
