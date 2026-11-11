"""Synthetic home health patients with internally consistent OASIS answers.

SYNTHETIC DATA. No real patient information is used anywhere in this project.

How realism is anchored:
* Marginal response distributions for the PDGM functional items are calibrated to the
  "Percent of Periods with this Response Category" column of Table 8 in the CY 2026
  HH PPS final rule (CMS, CY 2024 claims). Where CMS reports a merged bin
  (e.g. "0 or 1"), the split inside the bin is an assumption of this project.
* Items are driven by correlated latent traits (mobility, self-care, cognition,
  respiratory, pain) through a Gaussian copula, so answers co-vary the way they do in
  real assessments (a patient who cannot transfer usually cannot bathe in a tub).
* Hard clinical-coherence constraints (bedfast <-> M1850 4/5 etc.) are enforced so the
  ground truth always passes the QA consistency rules in `qa_rules.py`.
* The clinical-group mix, admission source/timing and comorbidity rates are
  illustrative assumptions, not CMS statistics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import norm

# Target marginals (response -> probability). Merged-bin totals are from CMS Table 8.
MARGINALS = {
    "M1800": {"0": 0.120, "1": 0.112, "2": 0.620, "3": 0.148},          # 0/1 = 23.2%
    "M1810": {"0": 0.095, "1": 0.080, "2": 0.700, "3": 0.125},          # 0/1 = 17.5%
    "M1820": {"0": 0.045, "1": 0.040, "2": 0.641, "3": 0.274},          # 0/1 = 8.5%
    "M1830": {"0": 0.008, "1": 0.014, "2": 0.094, "3": 0.330, "4": 0.157,
              "5": 0.300, "6": 0.098},                                  # 3/4 = 48.7%, 5/6 = 39.8%
    "M1840": {"0": 0.300, "1": 0.294, "2": 0.300, "3": 0.026, "4": 0.080},  # 0/1 = 59.4%
    "M1850": {"0": 0.010, "1": 0.173, "2": 0.700, "3": 0.090, "4": 0.017, "5": 0.010},  # 2-5 = 81.7%
    "M1860": {"0": 0.010, "1": 0.018, "2": 0.131, "3": 0.662, "4": 0.090,
              "5": 0.061, "6": 0.027},                                  # 4-6 = 17.8%
}
TARGET_M1033_4PLUS = 0.436  # Table 8: share of periods with >= 4 of responses 1-7 marked

# Which latent trait drives each item, and how strongly (rest is item-specific noise).
LOADINGS = {
    "M1800": ("selfcare", 0.80), "M1810": ("selfcare", 0.85), "M1820": ("selfcare", 0.85),
    "M1830": ("selfcare", 0.80), "M1840": ("mobility", 0.80), "M1850": ("mobility", 0.90),
    "M1860": ("mobility", 0.92),
}

CLINICAL_GROUP_MIX = {  # illustrative
    "MMTA - Cardiac": 0.17, "MS Rehab": 0.16, "MMTA - Other": 0.10, "Neuro": 0.10,
    "Wound": 0.10, "MMTA - Surgical Aftercare": 0.06, "MMTA - Respiratory": 0.08,
    "MMTA - Endocrine": 0.06, "Complex": 0.04, "MMTA - GI/GU": 0.05,
    "MMTA - Infectious": 0.05, "Behavioral Health": 0.03,
}
PATIENT_NAMES = ["Mrs. Alvarez", "Mr. Okafor", "Mrs. Chen", "Mr. Kowalski", "Mrs. Patel",
                 "Mr. Johnson", "Mrs. Nguyen", "Mr. Rossi", "Mrs. Washington", "Mr. Haddad",
                 "Mrs. Murphy", "Mr. Silva", "Mrs. Kim", "Mr. Brennan", "Mrs. Duarte"]
CAREGIVERS = ["daughter", "son", "husband", "wife", "granddaughter", "neighbor"]


@dataclass
class Patient:
    patient_id: str
    name: str
    sex: str
    caregiver: str | None
    clinical_group: str
    timing: str
    admission_source: str
    comorbidity: int
    latent: dict
    answers: dict


def _cut(z: float, marg: dict[str, float]) -> str:
    """Map a standard-normal draw to an ordinal response using cumulative marginals."""
    u = norm.cdf(z)
    acc = 0.0
    for code, p in marg.items():
        acc += p
        if u <= acc:
            return code
    return list(marg)[-1]


def _gg_from(rng, table: dict[str, list[tuple[str, float]]], key: str) -> str:
    opts = table[key]
    codes, probs = zip(*opts)
    return str(rng.choice(codes, p=np.array(probs) / sum(probs)))


# GG performance codes conditional on the closest M item (clinically plausible mappings).
GG_FROM_M1850 = {
    "GG0170D": {"0": [("06", 1)], "1": [("05", .4), ("04", .6)], "2": [("04", .3), ("03", .7)],
                "3": [("02", .7), ("01", .3)], "4": [("88", 1)], "5": [("88", 1)]},
    "GG0170E": {"0": [("06", 1)], "1": [("05", .3), ("04", .7)], "2": [("04", .2), ("03", .8)],
                "3": [("02", .6), ("01", .4)], "4": [("01", .5), ("88", .5)], "5": [("01", 1)]},
    "GG0170C": {"0": [("06", 1)], "1": [("06", .5), ("05", .5)], "2": [("05", .3), ("04", .7)],
                "3": [("03", .6), ("02", .4)], "4": [("03", .5), ("02", .5)], "5": [("01", 1)]},
}
GG0170I_FROM_M1860 = {"0": [("06", 1)], "1": [("06", .8), ("05", .2)], "2": [("05", .3), ("04", .7)],
                      "3": [("04", .6), ("03", .4)], "4": [("88", 1)], "5": [("88", 1)], "6": [("88", 1)]}
GG0130B_FROM_M1800 = {"0": [("06", 1)], "1": [("05", 1)], "2": [("04", .6), ("03", .4)],
                      "3": [("02", .6), ("01", .4)]}
GG0130C_FROM_M1840 = {"0": [("06", .7), ("05", .3)], "1": [("04", 1)], "2": [("04", .3), ("03", .7)],
                      "3": [("03", .5), ("02", .5)], "4": [("02", .3), ("01", .7)]}


def _enforce_coherence(a: dict) -> None:
    # Bedfast is a shared state across M1850 and M1860.
    if a["M1860"] == "6" and a["M1850"] not in ("4", "5"):
        a["M1850"] = "4"
    if a["M1850"] in ("4", "5") and a["M1860"] != "6":
        a["M1850"] = "3"
    # Chairfast or bedfast patients cannot shower/tub-bathe independently.
    if a["M1860"] in ("4", "5", "6") and a["M1830"] in ("0", "1", "2", "3"):
        a["M1830"] = "5" if a["M1860"] != "6" else "6"
    # Independent transfer & ambulation patients don't need a bedpan.
    if a["M1850"] == "0" and a["M1840"] in ("3", "4"):
        a["M1840"] = "1"
    # Bedfast patients can't walk to the toilet.
    if a["M1860"] == "6" and a["M1840"] in ("0", "1"):
        a["M1840"] = "3"


def _m1033(rng, frailty: float, cognition: float, n_meds_high: bool, base: float) -> list[str]:
    probs = {
        "1": 0.30 + 0.25 * frailty, "2": 0.20 + 0.15 * frailty, "3": 0.30 + 0.20 * frailty,
        "4": 0.25 + 0.15 * frailty, "5": 0.12 + 0.30 * cognition, "6": 0.10 + 0.25 * cognition,
        "7": 0.85 if n_meds_high else 0.25, "8": 0.20 + 0.20 * frailty, "9": 0.05,
    }
    marked = [k for k, p in probs.items() if rng.random() < min(0.97, p * base)]
    return marked or ["10"]


def generate_patients(n: int, seed: int = 7, m1033_scale: float = 1.31) -> list[Patient]:
    rng = np.random.default_rng(seed)
    groups, gprobs = zip(*CLINICAL_GROUP_MIX.items())
    corr = np.array([  # mobility, selfcare, cognition, respiratory, pain
        [1.0, 0.70, 0.30, 0.25, 0.20],
        [0.70, 1.0, 0.45, 0.25, 0.15],
        [0.30, 0.45, 1.0, 0.05, 0.00],
        [0.25, 0.25, 0.05, 1.0, 0.05],
        [0.20, 0.15, 0.00, 0.05, 1.0],
    ])
    chol = np.linalg.cholesky(corr)
    out = []
    for i in range(n):
        group = str(rng.choice(groups, p=np.array(gprobs) / sum(gprobs)))
        z = chol @ rng.standard_normal(5)
        mob, self_, cog, resp, pain = z
        if group in ("MMTA - Respiratory", "MMTA - Cardiac"):
            resp += 0.9
        if group in ("MS Rehab", "MMTA - Surgical Aftercare", "Wound"):
            pain += 0.6
        if group in ("Neuro", "Behavioral Health"):
            cog += 0.7
        latent = {"mobility": mob, "selfcare": self_}
        a: dict = {}
        for item, (trait, load) in LOADINGS.items():
            zi = load * latent[trait] + np.sqrt(1 - load**2) * rng.standard_normal()
            a[item] = _cut(zi, MARGINALS[item])
        _enforce_coherence(a)

        # Non-PDGM items
        a["M1400"] = _cut(0.8 * resp + 0.6 * rng.standard_normal(),
                          {"0": .30, "1": .32, "2": .25, "3": .10, "4": .03})
        if rng.random() < 0.03:
            a["J0510"] = "8"
        else:
            a["J0510"] = _cut(0.8 * pain + 0.6 * rng.standard_normal(),
                              {"0": .22, "1": .28, "2": .27, "3": .16, "4": .07})
        if rng.random() < 0.03:
            a["M2020"] = "NA"
        else:
            a["M2020"] = _cut(0.8 * cog + 0.6 * rng.standard_normal(),
                              {"0": .38, "1": .30, "2": .17, "3": .15})
        frailty = float(norm.cdf(0.6 * mob + 0.4 * self_))
        a["M1033"] = _m1033(rng, frailty, float(norm.cdf(cog)), rng.random() < 0.7, m1033_scale)

        # Section GG performance, conditional on M items (keeps the two sections coherent)
        for gg, table in GG_FROM_M1850.items():
            a[gg] = _gg_from(rng, table, a["M1850"])
        a["GG0170I"] = _gg_from(rng, GG0170I_FROM_M1860, a["M1860"])
        a["GG0130B"] = _gg_from(rng, GG0130B_FROM_M1800, a["M1800"])
        a["GG0130C"] = _gg_from(rng, GG0130C_FROM_M1840, a["M1840"])
        a["GG0130A"] = "06" if self_ < 0.3 else ("05" if self_ < 1.3 else "04")
        if a["M1800"] == "3":
            a["GG0130A"] = "03"

        timing = "Early" if rng.random() < 0.35 else "Late"
        source = "Institutional" if rng.random() < (0.55 if timing == "Early" else 0.12) else "Community"
        comorb = int(rng.choice([0, 1, 2], p=[0.45, 0.40, 0.15]))
        sex = "F" if rng.random() < 0.62 else "M"
        name = str(rng.choice([n for n in PATIENT_NAMES if n.startswith("Mrs" if sex == "F" else "Mr.")]))
        caregiver = str(rng.choice(CAREGIVERS)) if rng.random() < 0.65 else None
        out.append(Patient(f"SYN-{i:04d}", name, sex, caregiver, group, timing, source, comorb,
                           {k: round(float(v), 3) for k, v in zip(
                               ["mobility", "selfcare", "cognition", "respiratory", "pain"],
                               [mob, self_, cog, resp, pain])}, a))
    return out


def patient_dict(p: Patient) -> dict:
    return asdict(p)
