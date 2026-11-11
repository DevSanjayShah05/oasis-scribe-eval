"""Second Read dashboard. Run:  streamlit run dashboard/app.py"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES, SYN = ROOT / "data" / "results", ROOT / "data" / "synthetic"

# Fixed categorical order (validated reference palette): colour follows the system, never its rank.
SYSTEM_ORDER = ["keyword_baseline", "qwen2.5:7b", "qwen2.5:7b+guard",
                "qwen2.5:7b+segmented+rag", "qwen2.5:7b+segmented+rag+guard"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
# Outcome colours use the reserved status palette, always paired with a text label.
OUTCOME_COLORS = {"correct": "#0ca30c", "abstained": "#b4b2a9", "wrong": "#d03b3b",
                  "correct_abstain": "#86b6ef", "hallucinated": "#ec835a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e7e6e1"

st.set_page_config(page_title="Second Read", layout="wide")


@st.cache_data
def load():
    summary = json.loads((RES / "summary.json").read_text())
    dec = pd.read_csv(RES / "decisions.csv", keep_default_na=False, na_values=[])
    items = pd.read_csv(RES / "item_metrics.csv")
    brk = pd.read_csv(RES / "breakdown.csv")
    pdg = pd.read_csv(RES / "pdgm_impact.csv")
    qa = pd.read_csv(RES / "qa_rule_metrics.csv")
    tx = {json.loads(l)["transcript_id"]: json.loads(l) for l in open(SYN / "transcripts.jsonl")}
    preds = {}
    for f in RES.glob("predictions_*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            preds.setdefault(r["system"], {})[r["transcript_id"]] = r["prediction"]
    return summary, dec, items, brk, pdg, qa, tx, preds


def style_fig(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=36, b=8), plot_bgcolor="#fcfcfb",
                      paper_bgcolor="#fcfcfb", font=dict(color=INK2, size=13),
                      legend=dict(orientation="h", y=-0.2), hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


summary, dec, items, brk, pdg, qa, tx, preds = load()
names = summary["pretty_names"]
systems = [s for s in SYSTEM_ORDER if s in summary["systems"]]
color = {s: SERIES[SYSTEM_ORDER.index(s)] for s in systems}
llm_ids = {t for t, r in tx.items() if r["llm_eval_subset"]}

st.title("Second Read")
st.caption("An evaluation harness for AI scribes that write OASIS. **All transcripts and patients are "
           "SYNTHETIC.** CMS reference data: OASIS-E2 Guidance Manual (04/2026), CY 2026 HH PPS final rule.")

tabs = st.tabs(["Overview", "Difficulty & style", "Items", "PDGM impact", "QA rules", "Transcript explorer"])

with tabs[0]:
    st.subheader(f"Head-to-head on the {summary['eval_subset_transcripts']}-transcript eval set")
    rows = []
    for s in systems:
        m = summary["systems"][s]["eval_subset"]
        rows.append({"System": names[s], "Accuracy (answerable)": m["accuracy_answerable"],
                     "95% CI": f"{m['accuracy_ci95'][0]:.3f}-{m['accuracy_ci95'][1]:.3f}",
                     "Wrong answers": m["wrong_rate_answerable"], "Left blank": m["abstain_rate_answerable"],
                     "Hallucination (undiscussed items)": m["hallucination_rate"],
                     "PDGM level mismatch": m.get("pdgm_level_mismatch_rate"),
                     "Mean |$ error| / period": m.get("pdgm_mean_abs_payment_error"),
                     "QA flag precision": m.get("qa_flag_precision"),
                     "Median latency (s)": m.get("median_latency_s")})
    table = pd.DataFrame(rows)
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                                ["Accuracy (answerable)", "Wrong answers", "Left blank",
                                 "Hallucination (undiscussed items)", "PDGM level mismatch", "QA flag precision"]}
                 | {"Mean |$ error| / period": st.column_config.NumberColumn(format="$%.2f")})

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for s in systems:
            m = summary["systems"][s]["eval_subset"]
            acc, (lo, hi) = m["accuracy_answerable"], m["accuracy_ci95"]
            fig.add_bar(y=[names[s]], x=[acc], orientation="h", marker_color=color[s], name=names[s],
                        error_x=dict(type="data", symmetric=False, array=[hi - acc], arrayminus=[acc - lo],
                                     color=INK2, thickness=1.5),
                        hovertemplate=f"{names[s]}<br>accuracy %{{x:.1%}}<br>95% CI {lo:.1%}-{hi:.1%}<extra></extra>",
                        showlegend=False)
        fig.update_layout(title="Accuracy on answerable items (95% cluster-bootstrap CI)",
                          xaxis_tickformat=".0%", xaxis_range=[0, 1], bargap=0.35)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(fig), width="stretch")
    with c2:
        d = dec[dec.transcript_id.isin(llm_ids)]
        mix = d.groupby(["system", "outcome"]).size().unstack(fill_value=0)
        mix = mix.div(mix.sum(axis=1), axis=0)
        fig = go.Figure()
        for o in ["correct", "abstained", "wrong", "correct_abstain", "hallucinated"]:
            if o in mix:
                fig.add_bar(y=[names[s] for s in systems], x=[mix.loc[s, o] for s in systems], orientation="h",
                            name=o.replace("_", " "), marker_color=OUTCOME_COLORS[o],
                            marker_line=dict(color="#fcfcfb", width=2),
                            hovertemplate="%{y}<br>" + o.replace("_", " ") + ": %{x:.1%}<extra></extra>")
        fig.update_layout(barmode="stack", title="Every item decision, by outcome", xaxis_tickformat=".0%")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(fig), width="stretch")

with tabs[1]:
    dim = st.radio("Break down by", ["difficulty", "style"], horizontal=True,
                   format_func=lambda x: "Difficulty tier" if x == "difficulty" else "Evidence style")
    b = brk[brk.dimension == dim]
    order = ["easy", "medium", "hard"] if dim == "difficulty" else \
        ["direct", "observed", "caregiver", "history", "correction", "temporal", "omitted"]
    metric = st.selectbox("Metric", ["accuracy_answerable", "wrong_rate_answerable", "abstain_rate_answerable",
                                     "hallucination_rate"])
    fig = go.Figure()
    for s in systems:
        bs = b[b.system == s].set_index("level").reindex([o for o in order if o in set(b.level)])
        fig.add_bar(x=bs.index, y=bs[metric], name=names[s], marker_color=color[s],
                    hovertemplate=names[s] + "<br>%{x}: %{y:.1%}<extra></extra>")
    fig.update_layout(barmode="group", yaxis_tickformat=".0%", title=metric.replace("_", " "), bargap=0.25,
                      bargroupgap=0.08)
    st.plotly_chart(style_fig(fig, 420), width="stretch")
    st.caption("'omitted' rows have no answerable items, so only hallucination is defined there.")
    st.dataframe(b.pivot_table(index="level", columns="system", values=metric).reindex(
        [o for o in order if o in set(b.level)]).rename(columns=names), width="stretch")

with tabs[2]:
    s = st.selectbox("System", systems, index=len(systems) - 1, format_func=lambda x: names[x])
    it = items[items.system == s].copy()
    it["label"] = it.item + " " + it.item_name.str.slice(0, 32)
    it = it.sort_values("accuracy_answerable")
    fig = go.Figure(go.Bar(y=it.label, x=it.accuracy_answerable, orientation="h", marker_color=color[s],
                           customdata=it[["wrong_rate_answerable", "quadratic_kappa"]].fillna(-1),
                           hovertemplate="%{y}<br>accuracy %{x:.1%}<br>wrong %{customdata[0]:.1%}"
                                         "<br>weighted kappa %{customdata[1]:.2f}<extra></extra>"))
    fig.update_layout(title=f"Accuracy by item - {names[s]}", xaxis_tickformat=".0%", xaxis_range=[0, 1])
    st.plotly_chart(style_fig(fig, 560), width="stretch")
    st.dataframe(it.drop(columns=["label", "system"]), hide_index=True, width="stretch")

with tabs[3]:
    st.markdown("Each scored transcript is priced twice with the CY 2026 national 30-day rate ($2,038.22 x "
                "case-mix weight): once with the true OASIS, once with the scribe's. Blank items are assumed to "
                "be filled in correctly by the clinician, so only wrong answers move money.")
    p = pdg[pdg.transcript_id.isin(llm_ids)]
    fig = go.Figure()
    for s in systems:
        ps = p[p.system == s]
        fig.add_box(x=ps.payment_delta, name=names[s], marker_color=color[s], boxpoints="all", jitter=0.4,
                    pointpos=0, line_width=1.5, marker_size=8,
                    hovertemplate="%{x:$,.0f}<extra>" + names[s] + "</extra>")
    fig.update_layout(title="Payment error per 30-day period (scribe - truth)", xaxis_tickprefix="$",
                      showlegend=False)
    st.plotly_chart(style_fig(fig, 420), width="stretch")
    agg = p.groupby("system").agg(periods=("transcript_id", "count"),
                                  level_wrong=("pred_level", lambda x: None),
                                  ).drop(columns="level_wrong")
    agg["level mismatch"] = p.assign(m=p.true_level != p.pred_level).groupby("system").m.mean()
    agg["over-coded periods"] = p.assign(m=p.payment_delta > 0).groupby("system").m.sum()
    agg["under-coded periods"] = p.assign(m=p.payment_delta < 0).groupby("system").m.sum()
    agg["net $ across periods"] = p.groupby("system").payment_delta.sum()
    st.dataframe(agg.rename(index=names), width="stretch")

with tabs[4]:
    s = st.selectbox("System ", systems, index=len(systems) - 1, format_func=lambda x: names[x])
    q = qa[qa.system == s][["rule_id", "description", "evaluable", "fired", "precision"]]
    st.dataframe(q, hide_index=True, width="stretch",
                 column_config={"precision": st.column_config.NumberColumn(
                     "precision (flag points at a real error)", format="%.2f")})
    m = summary["systems"][s]["eval_subset"]
    st.markdown(f"**{m['qa_flags']}** flags; precision **{(m['qa_flag_precision'] or 0):.0%}**; "
                f"share of wrong answers on rule-covered items that a rule caught: "
                f"**{(m['qa_error_recall'] or 0):.0%}**. Ground truth triggers 0 flags by construction.")

with tabs[5]:
    tid = st.selectbox("Transcript (eval set)", sorted(llm_ids),
                       format_func=lambda t: f"{t} - {tx[t]['difficulty']} - {tx[t]['clinical_group']}")
    t = tx[tid]
    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"**SYNTHETIC transcript** - {t['difficulty']} tier")
        st.text_area("transcript", t["text"], height=620, label_visibility="collapsed")
    with right:
        rows = []
        for code, e in t["expected"].items():
            r = {"item": code, "style": t["styles"][code], "truth": json.dumps(e)}
            for s in systems:
                base = s.replace("+guard", "")
                if tid in preds.get(base, {}):
                    v = ((preds[base][tid] or {}).get(code) or {}).get("value")
                    if s.endswith("+guard"):
                        dd = dec[(dec.system == s) & (dec.transcript_id == tid) & (dec.item == code)]
                        v = json.loads(dd.predicted.iloc[0]) if len(dd) else v
                    r[names[s]] = json.dumps(v)
            rows.append(r)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=660)
