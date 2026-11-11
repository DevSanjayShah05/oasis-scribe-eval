"""Segment-then-extract pipeline: one small, grounded LLM call per OASIS item.

1. Segment: score every transcript line against the item's description (TF-IDF cosine)
   and keep the best-matching lines plus a little context.
2. Retrieve: pull the item's own CMS guidance passages most similar to that segment.
3. Extract: ask the model for a single item, with only that segment and that guidance.

Compared with a single 18-item pass, each call sees ~10x less text, which is the main
lever for a small local model.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

from ..items import GG_ITEMS, ITEM_BY_CODE, ITEMS
from ..retrieval import _index
from .base import Prediction, sanitize
from .llm import OLLAMA_URL

GG_NAMES = dict(GG_ITEMS)
ITEM_QUERY_EXTRA = {
    "M1800": "grooming wash face brush teeth comb hair shave",
    "M1810": "dress upper body shirt sleeve buttons",
    "M1820": "dress lower body pants socks shoes",
    "M1830": "bathing shower tub sponge bath",
    "M1840": "toilet commode bedpan bathroom",
    "M1850": "transfer bed chair lift bear weight pivot bedfast turn",
    "M1860": "walk ambulation cane walker wheelchair stairs bedfast",
    "M1033": "falls hospital emergency room weight lost confused medications pills exhausted",
    "M1400": "short of breath winded breathing",
    "J0510": "pain sleep nights",
    "M2020": "pills medications pill box remind injections",
    "GG0130A": "eating feed food",
    "GG0130B": "oral hygiene brushing teeth dentures toothpaste",
    "GG0130C": "toileting hygiene clean after toilet wipes pants",
    "GG0170C": "lying to sitting side of bed",
    "GG0170D": "sit to stand stand up from chair",
    "GG0170E": "bed to chair transfer",
    "GG0170I": "walk ten feet doorway",
}

RULES = """You code ONE OASIS-E2 item from an excerpt of a home health start-of-care visit transcript.
- Code the patient's ability on the day of assessment ("now"), not before the hospital.
- If the patient's claim is contradicted by a caregiver or by what the nurse observed, use the observation.
- Statements about other people are not about the patient.
- If the excerpt has no evidence for this item, value must be null. Never guess.
- evidence = a short VERBATIM quote from the excerpt (max 25 words)."""


def segment(lines: list[str], code: str, top: int = 2, before: int = 1, after: int = 3) -> list[str]:
    item = ITEM_BY_CODE[code]
    query = f"{GG_NAMES.get(code, item.name)} {ITEM_QUERY_EXTRA[code]}"
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    mat = vec.fit_transform(lines + [query])
    sims = (mat[:-1] @ mat[-1].T).toarray().ravel()
    keep: set[int] = set()
    for i in np.argsort(-sims)[:top]:
        if sims[i] <= 0:
            continue
        keep.update(range(max(0, i - before), min(len(lines), i + after + 1)))
    return [lines[i] for i in sorted(keep)]


@lru_cache(maxsize=None)
def _item_chunk_ids(code: str) -> list[int]:
    chunks, _, _ = _index()
    key = code if code in GG_NAMES or code.startswith(("M", "J")) else code
    return [i for i, c in enumerate(chunks) if c["item"] == key]


def item_guidance(code: str, excerpt: str, k: int = 2, max_chars: int = 900) -> list[dict]:
    chunks, vec, mat = _index()
    ids = _item_chunk_ids(code)
    if code.startswith("GG"):
        ids = ids + [i for i, c in enumerate(chunks) if c["item"] == "GG_GENERAL"]
    if not ids:
        return []
    q = vec.transform([excerpt])
    scores = (mat[ids] @ q.T).toarray().ravel()
    order = [ids[j] for j in np.argsort(-scores)[:k]]
    return [{**chunks[i], "text": chunks[i]["text"][:max_chars]} for i in order]


class SegmentedExtractor:
    def __init__(self, model: str = "qwen2.5:7b", use_rag: bool = True, timeout: int = 300, workers: int = 4):
        self.model, self.use_rag, self.timeout, self.workers = model, use_rag, timeout, workers
        self.name = f"{model}+segmented{'+rag' if use_rag else ''}"
        self.last_meta: dict = {}

    def _call(self, code: str, excerpt: str) -> tuple[dict, dict]:
        item = ITEM_BY_CODE[code]
        resp = "\n".join(f"  {k} = {v}" for k, v in item.responses.items())
        kind = ("Return value as a LIST of codes (check all that apply; [\"10\"] only if all risks were denied)."
                if item.kind == "checklist" else "Return value as ONE response code string.")
        guidance = item_guidance(code, excerpt) if self.use_rag else []
        sys = f"{RULES}\n\nITEM {code} - {GG_NAMES.get(code, item.name)}\nResponses:\n{resp}\n{kind}"
        if guidance:
            sys += "\n\nCMS OASIS-E2 Guidance Manual excerpts:\n" + "\n---\n".join(g["text"] for g in guidance)
        val_schema = ({"type": ["array", "null"], "items": {"type": "string"}} if item.kind == "checklist"
                      else {"type": ["string", "null"]})
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": sys},
                         {"role": "user", "content": "EXCERPT:\n" + excerpt}],
            "format": {"type": "object", "properties": {"value": val_schema, "evidence": {"type": ["string", "null"]}},
                       "required": ["value", "evidence"]},
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 4096, "seed": 7},
        }
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        try:
            out = json.loads(d["message"]["content"])
        except json.JSONDecodeError:
            out = {"value": None, "evidence": None}
        return out, {"prompt_tokens": d.get("prompt_eval_count", 0), "output_tokens": d.get("eval_count", 0),
                     "guidance_ids": [g["id"] for g in guidance]}

    def extract(self, transcript: str) -> Prediction:
        lines = [ln for ln in transcript.splitlines() if ln.strip()]
        raw, meta = {}, {"prompt_tokens": 0, "output_tokens": 0, "guidance_ids": {}}
        t0 = time.time()
        excerpts = {it.code: "\n".join(segment(lines, it.code)) for it in ITEMS}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:  # needs OLLAMA_NUM_PARALLEL >= workers
            results = dict(zip(excerpts, pool.map(lambda c: self._call(c, excerpts[c]), excerpts)))
        for code, (out, m) in results.items():
            raw[code] = out
            meta["prompt_tokens"] += m["prompt_tokens"] or 0
            meta["output_tokens"] += m["output_tokens"] or 0
            meta["guidance_ids"][code] = m["guidance_ids"]
        meta["latency_s"] = round(time.time() - t0, 2)
        meta["parse_ok"] = True
        self.last_meta = meta
        return sanitize(raw)
