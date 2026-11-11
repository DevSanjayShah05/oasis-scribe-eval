"""Evaluate every prediction file in data/results/ and write the result tables.

Outputs (data/results/):
  summary.json            headline metrics per system (+ bootstrap 95% CIs)
  decisions.csv           one row per (system, transcript, item) decision
  item_metrics.csv        accuracy / kappa / hallucination per system x item
  breakdown.csv           metrics per system x difficulty tier and x evidence style
  pdgm_impact.csv         per (system, transcript) functional level and payment vs truth
  qa_rule_metrics.csv     per system x rule: fires, precision against real errors
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import pdgm, qa_rules
from .extract.base import apply_grounding_guard
from .items import ITEM_BY_CODE, ITEM_CODES, PDGM_ITEMS
from .metrics import bootstrap_ci, jaccard, outcome, weighted_kappa, within_one

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "data" / "results"
SYN = ROOT / "data" / "synthetic"
PRETTY = {
    "keyword_baseline": "Keyword baseline",
    "qwen2.5:7b": "LLM single-pass",
    "qwen2.5:7b+guard": "LLM single-pass + grounding guard",
    "qwen2.5:7b+segmented+rag": "LLM segmented + RAG",
    "qwen2.5:7b+segmented+rag+guard": "LLM segmented + RAG + grounding guard",
}


def _load_predictions() -> dict[str, dict[str, dict]]:
    systems: dict[str, dict[str, dict]] = {}
    for f in sorted(RES.glob("predictions_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            systems.setdefault(r["system"], {})[r["transcript_id"]] = r
    return systems


def _decisions(system: str, preds: dict, transcripts: dict) -> list[dict]:
    rows = []
    for tid, rec in preds.items():
        t = transcripts[tid]
        p = rec["prediction"] or {}
        for code in ITEM_CODES:
            e = t["expected"][code]
            v = (p.get(code) or {}).get("value")
            rows.append({
                "system": system, "transcript_id": tid, "difficulty": t["difficulty"],
                "style": t["styles"][code], "item": code, "expected": json.dumps(e),
                "predicted": json.dumps(v), "outcome": outcome(e, v),
                "failed_call": rec.get("error") is not None,
            })
    return rows


def _rates(df: pd.DataFrame) -> dict:
    ans = df[df.outcome.isin(["correct", "wrong", "abstained"])]
    una = df[df.outcome.isin(["correct_abstain", "hallucinated"])]
    n_ans = len(ans)
    return {
        "decisions": int(len(df)),
        "accuracy_answerable": float((ans.outcome == "correct").mean()) if n_ans else None,
        "wrong_rate_answerable": float((ans.outcome == "wrong").mean()) if n_ans else None,
        "abstain_rate_answerable": float((ans.outcome == "abstained").mean()) if n_ans else None,
        "precision_when_answered": float((ans.outcome == "correct").sum() / max(1, (ans.outcome != "abstained").sum())),
        "hallucination_rate": float((una.outcome == "hallucinated").mean()) if len(una) else None,
        "n_unanswerable": int(len(una)),
    }


def _ci(df: pd.DataFrame, col_fn) -> tuple[float, float]:
    by_t = [col_fn(g).tolist() for _, g in df.groupby("transcript_id")]
    return bootstrap_ci(by_t)


def _pdgm_rows(system: str, preds: dict, transcripts: dict) -> list[dict]:
    """Functional level and payment, assuming a clinician fills any item the scribe left blank.

    Only transcripts where every PDGM item was discussed are scored, so the truth is
    fully determined by the conversation.
    """
    rows = []
    for tid, rec in preds.items():
        t = transcripts[tid]
        if any(t["expected"][c] is None for c in PDGM_ITEMS):
            continue
        truth = {c: t["expected"][c] for c in PDGM_ITEMS}
        pred = {}
        n_wrong = 0
        for c in PDGM_ITEMS:
            v = ((rec["prediction"] or {}).get(c) or {}).get("value")
            pred[c] = truth[c] if v is None else v
            n_wrong += int(v is not None and outcome(truth[c], v) == "wrong")
        args = (t["clinical_group"], t["timing"], t["admission_source"], t["comorbidity"])
        tp, pp = pdgm.period_payment(truth, *args), pdgm.period_payment(pred, *args)
        rows.append({
            "system": system, "transcript_id": tid, "difficulty": t["difficulty"],
            "clinical_group": t["clinical_group"], "pdgm_items_wrong": n_wrong,
            "true_points": tp.functional_points, "pred_points": pp.functional_points,
            "true_level": tp.functional_level, "pred_level": pp.functional_level,
            "true_hipps": tp.hipps, "pred_hipps": pp.hipps,
            "true_payment": tp.payment, "pred_payment": pp.payment,
            "payment_delta": round(pp.payment - tp.payment, 2),
        })
    return rows


def _qa_rows(system: str, preds: dict, transcripts: dict) -> list[dict]:
    rows = []
    for tid, rec in preds.items():
        t = transcripts[tid]
        answers = {c: ((rec["prediction"] or {}).get(c) or {}).get("value") for c in ITEM_CODES}
        fired = {f["rule_id"] for f in qa_rules.check(answers)}
        for r in qa_rules.RULES:
            involved_wrong = any(outcome(t["expected"][c], answers[c]) in ("wrong", "hallucinated")
                                 for c in r.items)
            evaluable = all(answers[c] is not None for c in r.items)
            rows.append({"system": system, "transcript_id": tid, "rule_id": r.rule_id,
                         "evaluable": evaluable, "fired": r.rule_id in fired,
                         "involved_item_wrong": involved_wrong})
    return rows


def run_all() -> dict:
    transcripts = {json.loads(l)["transcript_id"]: json.loads(l) for l in open(SYN / "transcripts.jsonl")}
    raw = _load_predictions()
    systems: dict[str, dict] = {}
    for name, preds in raw.items():
        systems[name] = preds
        if name != "keyword_baseline":  # keyword evidence is a window, so a guard is meaningless
            systems[name + "+guard"] = {
                tid: {**rec, "prediction": apply_grounding_guard(rec["prediction"] or {}, transcripts[tid]["text"])}
                for tid, rec in preds.items()}

    llm_ids = {tid for tid, t in transcripts.items() if t["llm_eval_subset"]}
    decisions, pdgm_rows, qa_rows = [], [], []
    for name, preds in systems.items():
        decisions += _decisions(name, preds, transcripts)
        pdgm_rows += _pdgm_rows(name, preds, transcripts)
        qa_rows += _qa_rows(name, preds, transcripts)
    dec = pd.DataFrame(decisions)
    pdg = pd.DataFrame(pdgm_rows)
    qa = pd.DataFrame(qa_rows)

    summary: dict = {"eval_subset_transcripts": len(llm_ids), "full_transcripts": len(transcripts),
                     "systems": {}}
    for name in systems:
        for scope, ids in (("eval_subset", llm_ids), ("all", set(transcripts))):
            d = dec[(dec.system == name) & dec.transcript_id.isin(ids)]
            if d.empty or (scope == "all" and d.transcript_id.nunique() == len(llm_ids)):
                continue
            s = _rates(d)
            ans = d[d.outcome.isin(["correct", "wrong", "abstained"])]
            s["accuracy_ci95"] = _ci(ans, lambda g: (g.outcome == "correct").astype(float))
            s["wrong_rate_ci95"] = _ci(ans, lambda g: (g.outcome == "wrong").astype(float))
            s["transcripts"] = int(d.transcript_id.nunique())
            p = pdg[(pdg.system == name) & pdg.transcript_id.isin(ids)]
            if len(p):
                s["pdgm_scored_transcripts"] = int(len(p))
                s["pdgm_level_mismatch_rate"] = float((p.true_level != p.pred_level).mean())
                s["pdgm_mean_abs_payment_error"] = float(p.payment_delta.abs().mean())
                s["pdgm_net_payment_bias"] = float(p.payment_delta.mean())
                s["pdgm_share_any_item_wrong"] = float((p.pdgm_items_wrong > 0).mean())
            q = qa[(qa.system == name) & qa.transcript_id.isin(ids)]
            fired = q[q.fired]
            s["qa_flags"] = int(len(fired))
            s["qa_flag_precision"] = float(fired.involved_item_wrong.mean()) if len(fired) else None
            errs = d[(d.outcome == "wrong") & d.item.isin({c for r in qa_rules.RULES for c in r.items})]
            caught = 0
            for _, e in errs.iterrows():
                hit = fired[(fired.transcript_id == e.transcript_id)]
                caught += int(any(e["item"] in qa_rules.RULE_BY_ID[r].items for r in hit.rule_id))
            s["qa_error_recall"] = float(caught / len(errs)) if len(errs) else None
            meta = [r.get("meta") or {} for tid, r in systems[name].items() if tid in ids]
            lat = [m.get("latency_s") for m in meta if m.get("latency_s")]
            if lat:
                s["median_latency_s"] = float(np.median(lat))
                s["failed_calls"] = int(sum(1 for tid, r in systems[name].items() if tid in ids and r.get("error")))
            summary["systems"].setdefault(name, {})[scope] = s

    # Item-level metrics (eval subset)
    item_rows = []
    sub = dec[dec.transcript_id.isin(llm_ids)]
    for (name, code), d in sub.groupby(["system", "item"]):
        s = _rates(d)
        pairs = [(json.loads(e), json.loads(p)) for e, p in zip(d.expected, d.predicted)]
        if ITEM_BY_CODE[code].kind == "checklist":
            js = [jaccard(e, p) for e, p in pairs if e is not None and p is not None]
            s["m1033_mean_jaccard"] = float(np.mean(js)) if js else None
            agree = [(len([x for x in e if x in "1234567" and x != "10"]) >= 4) ==
                     (len([x for x in p if x in "1234567" and x != "10"]) >= 4)
                     for e, p in pairs if e is not None and p is not None]
            s["m1033_pdgm_flag_agreement"] = float(np.mean(agree)) if agree else None
        else:
            s["quadratic_kappa"] = weighted_kappa(code, pairs)
            s["within_one"] = within_one(code, pairs)
        item_rows.append({"system": name, "item": code, "item_name": ITEM_BY_CODE[code].name,
                          "pdgm_item": ITEM_BY_CODE[code].pdgm, **s})

    breakdown = []
    for dim in ("difficulty", "style"):
        for (name, level), d in sub.groupby(["system", dim]):
            breakdown.append({"system": name, "dimension": dim, "level": level, **_rates(d)})

    RES.mkdir(parents=True, exist_ok=True)
    dec.to_csv(RES / "decisions.csv", index=False)
    pd.DataFrame(item_rows).to_csv(RES / "item_metrics.csv", index=False)
    pd.DataFrame(breakdown).to_csv(RES / "breakdown.csv", index=False)
    pdg.to_csv(RES / "pdgm_impact.csv", index=False)
    qa_summary = (qa[qa.transcript_id.isin(llm_ids)].groupby(["system", "rule_id"])
                  .apply(lambda g: pd.Series({
                      "evaluable": int(g.evaluable.sum()), "fired": int(g.fired.sum()),
                      "precision": float(g[g.fired].involved_item_wrong.mean()) if g.fired.any() else np.nan}),
                         include_groups=False).reset_index())
    qa_summary["description"] = qa_summary.rule_id.map(lambda r: qa_rules.RULE_BY_ID[r].description)
    qa_summary.to_csv(RES / "qa_rule_metrics.csv", index=False)
    summary["pretty_names"] = {k: PRETTY.get(k, k) for k in systems}
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))
    _print(summary)
    return summary


def _print(summary: dict) -> None:
    print(f"{'system':45s} {'scope':12s} {'acc':>6s} {'wrong':>6s} {'abst':>6s} {'halluc':>7s} "
          f"{'PDGMlvl':>8s} {'MAE$':>7s} {'QAprec':>7s}")
    for name, scopes in summary["systems"].items():
        for scope, s in scopes.items():
            f = lambda k: "  -  " if s.get(k) is None else f"{s[k]:.3f}"
            print(f"{PRETTY.get(name, name):45s} {scope:12s} {f('accuracy_answerable'):>6s} "
                  f"{f('wrong_rate_answerable'):>6s} {f('abstain_rate_answerable'):>6s} {f('hallucination_rate'):>7s} "
                  f"{f('pdgm_level_mismatch_rate'):>8s} {s.get('pdgm_mean_abs_payment_error', float('nan')):7.1f} "
                  f"{f('qa_flag_precision'):>7s}")
