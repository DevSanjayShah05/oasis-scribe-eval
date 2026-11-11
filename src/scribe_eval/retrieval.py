"""TF-IDF retrieval over passages of the CMS OASIS-E2 Guidance Manual."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

REF = Path(__file__).resolve().parents[2] / "data" / "reference"


@lru_cache(maxsize=1)
def _index():
    chunks = [json.loads(line) for line in open(REF / "oasis_guidance_chunks.jsonl")]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    mat = vec.fit_transform([c["text"] for c in chunks])
    return chunks, vec, mat


def retrieve(query: str, k: int = 6, max_chars: int = 700) -> list[dict]:
    chunks, vec, mat = _index()
    scores = (mat @ vec.transform([query]).T).toarray().ravel()
    top = np.argsort(-scores)[:k]
    return [{**chunks[i], "text": chunks[i]["text"][:max_chars], "score": float(scores[i])} for i in top]


def retrieve_for_transcript(transcript: str, k: int = 8) -> list[dict]:
    """Retrieve guidance for the hardest coding decisions in a transcript.

    Queries are built from the transcript lines that tend to need a rule lookup
    (help/assist/device/before-vs-now/refused/not attempted), so the retrieved passages
    are the coding instructions for the situations actually present in this visit.
    """
    lines = [ln for ln in transcript.splitlines()
             if any(w in ln.lower() for w in ("help", "assist", "steady", "walker", "cane", "lift",
                                              "before", "now", "set", "laid", "attempt", "safe", "remind"))]
    seen, out = set(), []
    for q in (lines or [transcript])[:40]:
        for c in retrieve(q, k=2):
            if c["id"] not in seen and c["score"] > 0.05:
                seen.add(c["id"])
                out.append(c)
    out.sort(key=lambda c: -c["score"])
    return out[:k]
