"""Build the committed reference files in data/reference/ from raw CMS downloads.

Raw inputs (data/raw/cms/, gitignored; fetch with scripts/fetch_cms.sh):
  * CY 2026 Final HH PDGM Case Mix Weights and LUPA Thresholds (CMS zip, CSV)
  * Final OASIS-E2 Guidance Manual (effective 04/01/2026), converted with `pdftotext -layout`

Values transcribed by hand from the CY 2026 HH PPS final rule (90 FR 55342,
FR Doc 2025-21767, published 2025-12-02), because CMS publishes these tables as images:
  * Table 8  - OASIS functional points
  * Table 9  - functional impairment level thresholds by clinical group
  * Table 14 - national standardized 30-day period payment ($2,038.22, with temporary adjustment)
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "cms"
REF = ROOT / "data" / "reference"

SOURCE_RULE = "CY 2026 HH PPS Final Rule, 90 FR 55342 (FR Doc 2025-21767), Dec 2, 2025"

# Table 8: final OASIS points table for CY 2026. Keys are OASIS response codes.
FUNCTIONAL_POINTS = {
    "M1800": {"0": 0, "1": 0, "2": 3, "3": 3},
    "M1810": {"0": 0, "1": 0, "2": 5, "3": 5},
    "M1820": {"0": 0, "1": 0, "2": 4, "3": 12},
    "M1830": {"0": 0, "1": 0, "2": 2, "3": 10, "4": 10, "5": 17, "6": 17},
    "M1840": {"0": 0, "1": 0, "2": 6, "3": 6, "4": 6},
    "M1850": {"0": 0, "1": 1, "2": 4, "3": 4, "4": 4, "5": 4},
    "M1860": {"0": 0, "1": 0, "2": 5, "3": 1, "4": 20, "5": 20, "6": 20},
}
M1033_RULE = {"count_responses": ["1", "2", "3", "4", "5", "6", "7"], "min_marked": 4, "points": 12}

# Table 9: thresholds -> (low_max, medium_max); high = > medium_max.
# Keys use the clinical-group labels as they appear in the CMS case-mix weight CSV.
FUNCTIONAL_THRESHOLDS = {
    "MMTA - Other": (30, 45),
    "Behavioral Health": (31, 46),
    "Complex": (31, 54),
    "MS Rehab": (31, 45),
    "Neuro": (34, 52),
    "Wound": (33, 52),
    "MMTA - Surgical Aftercare": (30, 42),
    "MMTA - Cardiac": (28, 43),
    "MMTA - Endocrine": (27, 41),
    "MMTA - GI/GU": (34, 48),
    "MMTA - Infectious": (32, 46),
    "MMTA - Respiratory": (33, 46),
}

NATIONAL_RATE_2026 = 2038.22  # Table 14, with the CY 2026 temporary adjustment (0.97000)

# OASIS manual sections used as the retrieval corpus.
SECTION_HEADERS = {
    "M1800": r"^M1800: Grooming",
    "M1810": r"^M1810: Current Ability to Dress Upper Body",
    "M1820": r"^M1820: Current Ability to Dress Lower Body",
    "M1830": r"^M1830: Bathing",
    "M1840": r"^M1840: Toilet Transferring",
    "M1850": r"^M1850: Transferring",
    "M1860": r"^M1860: Ambulation/Locomotion",
    "M1033": r"^M1033: Risk of Hospitalization",
    "J0510": r"^J0510: Pain Effect on Sleep",
    "M1400": r"^M1400: When is the patient dyspneic",
    "M2020": r"^M2020: Management of Oral Medications",
    "GG0130A": r"^GG0130A Eating",
    "GG0130B": r"^GG0130B Oral hygiene",
    "GG0130C": r"^GG0130C Toileting hygiene",
    "GG0170C": r"^GG0170C Lying to sitting on side of bed",
    "GG0170D": r"^GG0170D Sit to stand",
    "GG0170E": r"^GG0170E Chair/bed-to-chair transfer",
    "GG0170I": r"^GG0170I Walk 10 feet",
    "GG_GENERAL": r"^Response Specific Instructions . General",
}
ANY_HEADER = re.compile(r"^(M\d{4}|J\d{4}(-J\d{4})?|GG\d{4}[A-Z]{0,2}|[A-Z]\d{4})[:. ]")
FOOTER = re.compile(r"^(OASIS-E2|Effective \d\d/\d\d/\d{4}|Centers for Medicare|Chapter \d)")


def build_case_mix() -> list[dict]:
    src = RAW / "pdgm" / "CY 2026 Final HH PDGM Case Mix Weights and LUPA Thresholds.csv"
    rows = []
    with open(src, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            if not r or not r[0].strip():
                continue
            group, level = r[1].rsplit(" - ", 1)
            timing, source = [s.strip() for s in r[2].split(" - ")]
            rows.append({
                "hipps": r[0].strip(),
                "clinical_group": group.strip(),
                "functional_level": level.strip(),
                "timing": timing,
                "admission_source": source,
                "comorbidity": int(r[3]),
                "weight": float(r[4]),
                "lupa_threshold": int(r[5]),
            })
    return rows


def clean_lines(lines: list[str]) -> str:
    kept = [ln.rstrip() for ln in lines if ln.strip() and not FOOTER.match(ln.strip())]
    text = "\n".join(kept)
    text = re.sub(r"[ \t]{3,}", "  ", text)
    return text.strip()


def build_guidance() -> list[dict]:
    lines = (RAW / "manual.txt").read_text(encoding="utf-8").splitlines()
    chunks = []
    for item, pat in SECTION_HEADERS.items():
        rx = re.compile(pat)
        start = next(i for i, ln in enumerate(lines) if rx.match(ln.strip()))
        end = start + 1
        limit = 160 if item != "GG_GENERAL" else 90
        while end < len(lines) and end - start < limit:
            ln = lines[end].strip()
            if ANY_HEADER.match(ln) and not ln.startswith(item[:7]) and item != "GG_GENERAL":
                # stop at the next item's section (allow sub-mentions of the same item)
                if re.match(r"^(M\d{4}|J\d{4}|GG\d{4}[A-Z])[: ]", ln) and not ln.startswith(item):
                    break
            end += 1
        body = clean_lines(lines[start:end])
        # Split long sections into ~1,200 character passages for retrieval.
        paras, buf = [], ""
        for para in body.split("\n"):
            if len(buf) + len(para) > 1200 and buf:
                paras.append(buf.strip())
                buf = ""
            buf += para + "\n"
        if buf.strip():
            paras.append(buf.strip())
        for k, p in enumerate(paras):
            chunks.append({"id": f"{item}-{k}", "item": item, "text": p,
                           "source": "CMS OASIS-E2 Guidance Manual (eff. 04/01/2026)"})
    return chunks


def main() -> None:
    REF.mkdir(parents=True, exist_ok=True)
    case_mix = build_case_mix()
    assert len(case_mix) == 432, len(case_mix)
    with open(REF / "pdgm_case_mix_cy2026.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(case_mix[0]))
        w.writeheader()
        w.writerows(case_mix)
    groups = sorted({r["clinical_group"] for r in case_mix})
    assert set(groups) == set(FUNCTIONAL_THRESHOLDS), groups

    pdgm = {
        "source": SOURCE_RULE,
        "national_standardized_30_day_rate": NATIONAL_RATE_2026,
        "functional_points": FUNCTIONAL_POINTS,
        "m1033_rule": M1033_RULE,
        "functional_thresholds": {k: {"low_max": v[0], "medium_max": v[1]}
                                  for k, v in FUNCTIONAL_THRESHOLDS.items()},
    }
    (REF / "pdgm_cy2026.json").write_text(json.dumps(pdgm, indent=2))

    chunks = build_guidance()
    with open(REF / "oasis_guidance_chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from scribe_eval.items import catalog_as_dict
    (REF / "oasis_items.json").write_text(json.dumps(catalog_as_dict(), indent=2))
    print(f"case-mix rows={len(case_mix)} guidance chunks={len(chunks)} items={len(catalog_as_dict())}")


if __name__ == "__main__":
    main()
