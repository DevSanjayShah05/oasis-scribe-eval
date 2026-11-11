"""LLM extractor: transcript -> OASIS answers with verbatim evidence quotes (local Ollama)."""

from __future__ import annotations

import json
import os
import time

import requests

from ..items import ITEMS, ITEM_CODES
from ..retrieval import retrieve_for_transcript
from .base import Prediction, sanitize

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

SYSTEM = """You are an OASIS-E2 documentation assistant for a home health start-of-care visit.
Read the visit transcript and answer each OASIS item ONLY from what is said in the transcript.

Rules:
1. Code the patient's ability on the DAY OF ASSESSMENT. If the patient describes how things were
   before the hospital and how they are now, code NOW.
2. When the patient's claim is contradicted by the caregiver or by what the nurse observed, trust the
   observation / the more reliable account.
3. Statements about other people (a spouse, a sister, an in-law) are NOT about the patient.
4. If the transcript contains no evidence for an item, return null for "value". Never guess.
5. "evidence" must be a short quote copied VERBATIM from the transcript (max 25 words).
6. Section GG items use two-digit codes (06, 05, 04, 03, 02, 01, 07, 09, 10, 88).
7. M1033 is check-all-that-apply: return a list of response codes; ["10"] only if risks were asked about and all denied.

OASIS items and allowed responses:
"""


def _catalog_text() -> str:
    parts = []
    for it in ITEMS:
        resp = "; ".join(f"{k}={v}" for k, v in it.responses.items())
        parts.append(f"- {it.code} {it.name}: {resp}")
    return "\n".join(parts)


def _schema() -> dict:
    props = {}
    for it in ITEMS:
        val = ({"type": ["array", "null"], "items": {"type": "string"}} if it.kind == "checklist"
               else {"type": ["string", "null"]})
        props[it.code] = {"type": "object", "properties": {"value": val, "evidence": {"type": ["string", "null"]}},
                          "required": ["value", "evidence"]}
    return {"type": "object", "properties": props, "required": ITEM_CODES}


class OllamaExtractor:
    def __init__(self, model: str = "qwen2.5:7b", use_rag: bool = False, k: int = 8,
                 num_ctx: int = 12288, timeout: int = 600):
        self.model, self.use_rag, self.k = model, use_rag, k
        self.num_ctx, self.timeout = num_ctx, timeout
        self.name = f"{model}{'+rag' if use_rag else ''}"
        self.last_meta: dict = {}

    def build_prompt(self, transcript: str) -> tuple[str, list[dict]]:
        system = SYSTEM + _catalog_text()
        guidance = retrieve_for_transcript(transcript, k=self.k) if self.use_rag else []
        if guidance:
            system += "\n\nRelevant CMS OASIS-E2 Guidance Manual excerpts:\n" + "\n---\n".join(
                f"[{g['item']}] {g['text']}" for g in guidance)
        return system, guidance

    def extract(self, transcript: str) -> Prediction:
        system, guidance = self.build_prompt(transcript)
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": f"TRANSCRIPT:\n{transcript}\n\nReturn the JSON object."}],
            "format": _schema(),
            "stream": False,
            "options": {"temperature": 0, "num_ctx": self.num_ctx, "seed": 7},
        }
        t0 = time.time()
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        raw = data["message"]["content"]
        try:
            parsed = json.loads(raw)
            parse_ok = True
        except json.JSONDecodeError:
            parsed, parse_ok = {}, False
        self.last_meta = {
            "latency_s": round(time.time() - t0, 2),
            "prompt_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
            "parse_ok": parse_ok,
            "guidance_ids": [g["id"] for g in guidance],
        }
        return sanitize(parsed)
