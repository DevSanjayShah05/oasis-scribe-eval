"""Common extractor interface and the grounding guard."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Protocol

from ..items import ITEM_BY_CODE, ITEM_CODES

# Prediction format: {item_code: {"value": str | list[str] | None, "evidence": str | None}}
Prediction = dict[str, dict]


class Extractor(Protocol):
    name: str

    def extract(self, transcript: str) -> Prediction: ...


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(uh|um|you know|i mean|like),?", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _support_one(e: str, t: str) -> float:
    if not e:
        return 0.0
    if e in t:
        return 1.0
    m = SequenceMatcher(None, e, t, autojunk=False).find_longest_match(0, len(e), 0, len(t))
    return m.size / len(e)


def evidence_support(evidence: str | None, transcript: str) -> float:
    """How well the evidence is grounded in the transcript (0-1).

    A quote may stitch together several sentences from different turns (common for
    checklist items like M1033), so each sentence is checked on its own and the
    weakest sentence decides: one invented sentence is enough to fail the quote.
    """
    if not evidence:
        return 0.0
    t = _norm(transcript)
    whole = _support_one(_norm(evidence), t)
    if whole == 1.0:
        return 1.0
    parts = [_norm(p) for p in re.split(r"(?<=[.!?])\s+|\n|\s*\.\.\.\s*|\s+[A-Z]+( \([a-z ]+\))?:\s+", evidence) if p]
    parts = [p for p in parts if len(p.split()) >= 3]
    if not parts:
        return whole
    return max(whole, min(_support_one(p, t) for p in parts))


def sanitize(pred: dict) -> Prediction:
    """Coerce raw model output into valid OASIS codes; invalid answers become None."""
    out: Prediction = {}
    for code in ITEM_CODES:
        raw = pred.get(code) or {}
        if not isinstance(raw, dict):
            raw = {"value": raw, "evidence": None}
        val, ev = raw.get("value"), raw.get("evidence")
        item = ITEM_BY_CODE[code]
        if item.kind == "checklist":
            if val in (None, "", [], "null"):
                val = None
            else:
                vals = val if isinstance(val, list) else re.split(r"[,\s]+", str(val))
                vals = sorted({str(v).strip() for v in vals if str(v).strip() in item.responses}, key=int)
                val = vals or None
        else:
            if val is None or str(val).strip().lower() in ("", "null", "none", "unknown", "-"):
                val = None
            else:
                v = str(val).strip()
                if item.section == "GG" and v.isdigit() and len(v) == 1:
                    v = "0" + v
                val = v if v in item.responses else None
        out[code] = {"value": val, "evidence": ev if isinstance(ev, str) else None}
    return out


def apply_grounding_guard(pred: Prediction, transcript: str, threshold: float = 0.7) -> Prediction:
    """Abstain on any answer whose evidence quote cannot be found in the transcript."""
    guarded: Prediction = {}
    for code, p in pred.items():
        support = evidence_support(p.get("evidence"), transcript)
        if p.get("value") is not None and support < threshold:
            guarded[code] = {"value": None, "evidence": p.get("evidence"), "guarded": True}
        else:
            guarded[code] = dict(p)
    return guarded
