"""Command line entry point.

    python -m scribe_eval.cli generate --n 300
    python -m scribe_eval.cli extract --system keyword
    python -m scribe_eval.cli extract --system llm --model qwen2.5:7b [--rag] --subset llm
    python -m scribe_eval.cli evaluate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .synth.patients import generate_patients, patient_dict
from .synth.transcripts import make_transcript, transcript_dict, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "data" / "synthetic"
RES = ROOT / "data" / "results"
TIERS = ["easy", "medium", "hard"]


def load_transcripts(subset: str | None = None) -> list[dict]:
    rows = [json.loads(line) for line in open(SYN / "transcripts.jsonl")]
    if subset == "llm":
        rows = [r for r in rows if r["llm_eval_subset"]]
    return rows


def cmd_generate(n: int, seed: int, llm_per_tier: int) -> None:
    SYN.mkdir(parents=True, exist_ok=True)
    patients = generate_patients(n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    rows, counts = [], {t: 0 for t in TIERS}
    for i, p in enumerate(patients):
        tier = TIERS[i % 3]
        t = transcript_dict(make_transcript(p, tier, rng, f"TX-{i:04d}"))
        t["llm_eval_subset"] = counts[tier] < llm_per_tier
        counts[tier] += 1
        rows.append(t)
    write_jsonl([patient_dict(p) for p in patients], SYN / "patients.jsonl")
    write_jsonl(rows, SYN / "transcripts.jsonl")
    print(f"wrote {len(rows)} SYNTHETIC transcripts ({sum(r['llm_eval_subset'] for r in rows)} in LLM eval subset)")


def cmd_extract(system: str, model: str, rag: bool, subset: str | None, limit: int | None) -> None:
    from .extract.llm import OllamaExtractor
    from .extract.rules import KeywordExtractor
    from .extract.segmented import SegmentedExtractor

    RES.mkdir(parents=True, exist_ok=True)
    if system == "keyword":
        ext = KeywordExtractor()
    elif system == "segmented":
        ext = SegmentedExtractor(model=model, use_rag=rag)
    else:
        ext = OllamaExtractor(model=model, use_rag=rag)
    out_path = RES / f"predictions_{ext.name.replace(':', '-').replace('+', '_')}.jsonl"
    done = set()
    if out_path.exists():  # resumable: long local-LLM runs can be interrupted
        done = {json.loads(line)["transcript_id"] for line in open(out_path)}
    rows = load_transcripts(subset)
    if limit:
        rows = rows[:limit]
    todo = [r for r in rows if r["transcript_id"] not in done]
    print(f"{ext.name}: {len(done)} done, {len(todo)} to go -> {out_path.name}", flush=True)
    with open(out_path, "a") as f:
        for i, r in enumerate(todo, 1):
            try:
                pred = ext.extract(r["text"])
                meta = getattr(ext, "last_meta", {})
                err = None
            except Exception as e:  # keep going; failures are counted in evaluation
                pred, meta, err = {}, {}, repr(e)[:300]
            f.write(json.dumps({"transcript_id": r["transcript_id"], "system": ext.name,
                                "prediction": pred, "meta": meta, "error": err}) + "\n")
            f.flush()
            if i % 5 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} last latency={meta.get('latency_s')}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(prog="scribe_eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--n", type=int, default=300)
    g.add_argument("--seed", type=int, default=2026)
    g.add_argument("--llm-per-tier", type=int, default=20)
    e = sub.add_parser("extract")
    e.add_argument("--system", choices=["keyword", "llm", "segmented"], required=True)
    e.add_argument("--model", default="qwen2.5:7b")
    e.add_argument("--rag", action="store_true")
    e.add_argument("--subset", choices=["llm"], default=None)
    e.add_argument("--limit", type=int, default=None)
    sub.add_parser("evaluate")
    a = ap.parse_args()
    if a.cmd == "generate":
        cmd_generate(a.n, a.seed, a.llm_per_tier)
    elif a.cmd == "extract":
        cmd_extract(a.system, a.model, a.rag, a.subset, a.limit)
    elif a.cmd == "evaluate":
        from .evaluate import run_all
        run_all()


if __name__ == "__main__":
    main()
