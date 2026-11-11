"""Second Read API: extract OASIS answers from a transcript, QA them, and price the PDGM impact.

Run:  uvicorn api.main:app --reload   (from the repo root, with PYTHONPATH=src)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from scribe_eval import pdgm, qa_rules
from scribe_eval.extract.base import apply_grounding_guard
from scribe_eval.extract.rules import KeywordExtractor
from scribe_eval.items import catalog_as_dict

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="Second Read - OASIS scribe evaluation API", version="1.0.0",
              description="Synthetic-data research tool. Not for clinical or billing use.")


class PeriodContext(BaseModel):
    clinical_group: str = Field("MMTA - Cardiac", examples=["MMTA - Cardiac"])
    timing: Literal["Early", "Late"] = "Early"
    admission_source: Literal["Community", "Institutional"] = "Community"
    comorbidity: Literal[0, 1, 2] = 0


class ExtractRequest(BaseModel):
    transcript: str = Field(..., min_length=20)
    system: Literal["keyword", "llm", "segmented"] = "segmented"
    model: str = "qwen2.5:7b"
    grounding_guard: bool = True
    period: PeriodContext | None = None


class AnswersRequest(BaseModel):
    answers: dict = Field(..., examples=[{"M1800": "2", "M1810": "2", "M1820": "2", "M1830": "3",
                                          "M1840": "1", "M1850": "2", "M1860": "3", "M1033": ["1", "3", "7"]}])
    period: PeriodContext = PeriodContext()


def _pdgm_block(answers: dict, period: PeriodContext | None) -> dict | None:
    if period is None:
        return None
    if period.clinical_group not in pdgm.clinical_groups():
        raise HTTPException(422, f"clinical_group must be one of {pdgm.clinical_groups()}")
    pay = pdgm.period_payment(answers, period.clinical_group, period.timing, period.admission_source,
                              period.comorbidity)
    if pay is None:
        return {"complete": False, "note": "One or more PDGM functional items (M1800-M1860, M1033) is blank."}
    return {"complete": True, **pay.__dict__,
            "note": "National standardized CY 2026 rate x case-mix weight; not wage-index adjusted."}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/items")
def items() -> list[dict]:
    return catalog_as_dict()


@app.post("/extract")
def extract(req: ExtractRequest) -> dict:
    if req.system == "keyword":
        ext = KeywordExtractor()
    elif req.system == "llm":
        from scribe_eval.extract.llm import OllamaExtractor
        ext = OllamaExtractor(model=req.model)
    else:
        from scribe_eval.extract.segmented import SegmentedExtractor
        ext = SegmentedExtractor(model=req.model)
    try:
        pred = ext.extract(req.transcript)
    except Exception as e:  # Ollama not running, model missing, ...
        raise HTTPException(503, f"Extractor failed: {e!r}") from e
    if req.grounding_guard and req.system != "keyword":
        pred = apply_grounding_guard(pred, req.transcript)
    answers = {c: p.get("value") for c, p in pred.items()}
    return {"system": ext.name, "items": pred, "qa_flags": qa_rules.check(answers),
            "pdgm": _pdgm_block(answers, req.period), "meta": getattr(ext, "last_meta", {})}


@app.post("/qa-check")
def qa_check(req: AnswersRequest) -> dict:
    return {"flags": qa_rules.check(req.answers), "rules_checked": len(qa_rules.RULES)}


@app.post("/pdgm/functional")
def pdgm_functional(req: AnswersRequest) -> dict:
    return _pdgm_block(req.answers, req.period)


@app.get("/metrics")
def metrics() -> dict:
    path = ROOT / "data" / "results" / "summary.json"
    if not path.exists():
        raise HTTPException(404, "Run `python -m scribe_eval.cli evaluate` first.")
    return json.loads(path.read_text())
