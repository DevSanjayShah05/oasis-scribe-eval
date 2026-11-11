from collections import Counter

import numpy as np
import pytest

from scribe_eval import pdgm, qa_rules
from scribe_eval.extract.base import apply_grounding_guard, evidence_support, sanitize
from scribe_eval.extract.rules import KeywordExtractor
from scribe_eval.items import ITEM_BY_CODE, ITEM_CODES
from scribe_eval.metrics import outcome, weighted_kappa
from scribe_eval.synth.patients import MARGINALS, generate_patients
from scribe_eval.synth.transcripts import make_transcript

INDEPENDENT = {"M1800": "0", "M1810": "0", "M1820": "0", "M1830": "0", "M1840": "0",
               "M1850": "0", "M1860": "0", "M1033": ["10"]}


def test_catalog_has_18_items_with_valid_ordinals():
    assert len(ITEM_CODES) == 18
    for code in ITEM_CODES:
        item = ITEM_BY_CODE[code]
        assert set(item.ordinal) <= set(item.responses)


def test_functional_points_match_cms_table_8():
    assert pdgm.functional_points(INDEPENDENT) == 0
    worst = {"M1800": "3", "M1810": "3", "M1820": "3", "M1830": "6", "M1840": "4",
             "M1850": "5", "M1860": "6", "M1033": ["1", "2", "3", "4"]}
    assert pdgm.functional_points(worst) == 3 + 5 + 12 + 17 + 6 + 4 + 20 + 12
    # M1860 response 2 scores more than response 3 in CY 2026 (noted by CMS under Table 8)
    a2, a3 = dict(INDEPENDENT, M1860="2"), dict(INDEPENDENT, M1860="3")
    assert pdgm.functional_points(a2) == 5 and pdgm.functional_points(a3) == 1
    # M1033 counts only responses 1-7, needs four of them
    assert pdgm.functional_points(dict(INDEPENDENT, M1033=["1", "2", "3", "8", "9"])) == 0
    assert pdgm.functional_points(dict(INDEPENDENT, M1033=["1", "2", "3", "7"])) == 12
    assert pdgm.functional_points({"M1800": "0"}) is None


def test_thresholds_match_cms_table_9():
    assert pdgm.functional_level(30, "MMTA - Other") == "Low"
    assert pdgm.functional_level(31, "MMTA - Other") == "Medium"
    assert pdgm.functional_level(46, "MMTA - Other") == "High"
    assert pdgm.functional_level(54, "Complex") == "Medium"
    assert pdgm.functional_level(55, "Complex") == "High"
    assert len(pdgm.clinical_groups()) == 12


def test_payment_uses_cy2026_rate_and_case_mix_weight():
    p = pdgm.period_payment(INDEPENDENT, "Behavioral Health", "Early", "Community", 0)
    assert p.hipps == "1FA11"
    assert p.payment == pytest.approx(round(p.weight * 2038.22, 2))
    assert len(pdgm._case_mix()) == 432


def test_generator_is_deterministic_and_calibrated():
    a = generate_patients(50, seed=1)
    b = generate_patients(50, seed=1)
    assert [p.answers for p in a] == [p.answers for p in b]
    ps = generate_patients(4000, seed=5)
    for item, marg in MARGINALS.items():
        c = Counter(p.answers[item] for p in ps)
        for code, target in marg.items():
            assert abs(c[code] / len(ps) - target) < 0.04, (item, code)
    share_4plus = np.mean([len([x for x in p.answers["M1033"] if x in {"1", "2", "3", "4", "5", "6", "7"}]) >= 4
                           for p in ps])
    assert abs(share_4plus - 0.436) < 0.03


def test_ground_truth_passes_every_qa_rule():
    for p in generate_patients(3000, seed=9):
        assert qa_rules.check(p.answers) == [], p.answers


def test_qa_rules_fire_on_contradictions():
    a = {c: None for c in ITEM_CODES}
    a.update({"M1850": "4", "M1860": "2"})
    assert [f["rule_id"] for f in qa_rules.check(a)] == ["R01"]
    a = {c: None for c in ITEM_CODES}
    a.update({"GG0170I": "06", "M1860": "5"})
    assert "R07" in {f["rule_id"] for f in qa_rules.check(a)}


def test_transcript_omissions_are_unanswerable():
    rng = np.random.default_rng(0)
    for i, p in enumerate(generate_patients(60, seed=4)):
        t = make_transcript(p, "hard", rng, f"T{i}")
        for code, style in t.styles.items():
            assert (t.expected[code] is None) == (style == "omitted")
        assert all(t.expected[c] == p.answers[c] or t.expected[c] is None
                   for c in ITEM_CODES if c != "M1033")


def test_outcome_categories_and_kappa():
    assert outcome("2", "2") == "correct"
    assert outcome("2", "3") == "wrong"
    assert outcome("2", None) == "abstained"
    assert outcome(None, None) == "correct_abstain"
    assert outcome(None, "1") == "hallucinated"
    assert outcome(["1", "7"], ["7", "1"]) == "correct"
    pairs = [("0", "0"), ("1", "1"), ("2", "2"), ("3", "3"), ("2", "2"), ("1", "1")]
    assert weighted_kappa("M1800", pairs) == pytest.approx(1.0)


def test_sanitize_and_grounding_guard():
    raw = {"M1800": {"value": "2", "evidence": "my daughter has to help me brush my hair"},
           "GG0170D": {"value": "4", "evidence": "invented quote that is nowhere"},
           "M1033": {"value": ["7", "10x", "1"], "evidence": None}}
    p = sanitize(raw)
    assert p["GG0170D"]["value"] == "04"          # single digit GG code normalised
    assert p["M1033"]["value"] == ["1", "7"]      # invalid codes dropped, sorted
    assert p["M1810"]["value"] is None
    transcript = "PATIENT: Um, my daughter has to help me brush my hair, my arms won't go up."
    assert evidence_support(p["M1800"]["evidence"], transcript) == 1.0
    g = apply_grounding_guard(p, transcript)
    assert g["M1800"]["value"] == "2"
    assert g["GG0170D"]["value"] is None and g["GG0170D"]["guarded"]


def test_keyword_baseline_runs_on_easy_transcript():
    rng = np.random.default_rng(1)
    p = generate_patients(1, seed=2)[0]
    t = make_transcript(p, "easy", rng, "T0")
    pred = KeywordExtractor().extract(t.text)
    assert set(pred) == set(ITEM_CODES)
