"""Turn a synthetic patient into a synthetic home-visit transcript. SYNTHETIC DATA.

Each OASIS item is "expressed" in the conversation with one evidence style:

  direct      patient states their ability plainly
  observed    nurse narrates what they watched the patient do
  caregiver   a family caregiver describes the patient
  correction  patient overstates their ability; caregiver/nurse corrects it (truth = correction)
  temporal    patient contrasts before-hospital ability with now (truth = now, the day of assessment)
  omitted     item never comes up -> the correct scribe behaviour is to ABSTAIN (expected = None)

Difficulty tiers mix these styles and add noise:
  easy    direct/observed only, every item discussed, no noise
  medium  + caregiver reports, small talk, 6% of items omitted
  hard    + corrections, temporal contrasts, "other person" distractors, 12% omitted,
          speech-recognition-style disfluencies
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from ..items import GG_ITEMS, ITEM_BY_CODE, ITEM_CODES
from . import phrases as ph
from .patients import Patient

TIERS = {
    "easy":   {"styles": {"direct": .5, "observed": .5}, "omit": 0.0, "noise": 0.0, "small_talk": 2},
    "medium": {"styles": {"direct": .35, "observed": .3, "caregiver": .35}, "omit": .06, "noise": .05,
               "small_talk": 5},
    "hard":   {"styles": {"direct": .15, "observed": .15, "caregiver": .2, "correction": .3, "temporal": .2},
               "omit": .12, "noise": .18, "small_talk": 7},
}
FILLERS = ["uh", "um", "you know", "I mean", "like"]


@dataclass
class Transcript:
    transcript_id: str
    patient_id: str
    difficulty: str
    clinical_group: str
    timing: str
    admission_source: str
    comorbidity: int
    turns: list[dict]
    expected: dict
    styles: dict
    true_answers: dict

    @property
    def text(self) -> str:
        return render(self.turns)


def render(turns: list[dict]) -> str:
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)


class _Ctx:
    def __init__(self, p: Patient, rng: np.random.Generator):
        self.p, self.rng = p, rng
        f = p.sex == "F"
        self.tok = {"S": "She" if f else "He", "s": "she" if f else "he", "o": "her" if f else "him",
                    "p": "her" if f else "his", "self": "herself" if f else "himself",
                    "cg": p.caregiver or "aide", "name": p.name}
        self.cg_label = f"CAREGIVER ({p.caregiver})" if p.caregiver else None
        cg = p.caregiver or ""
        if cg in ("daughter", "son"):
            self.addr = "Mom" if f else "Dad"
        elif cg == "granddaughter":
            self.addr = "Grandma" if f else "Grandpa"
        elif cg in ("husband", "wife"):
            self.addr = "Honey"
        else:
            self.addr = p.name

    def fmt(self, s: str, **extra) -> str:
        return s.format(**self.tok, **extra)

    def pick(self, seq):
        return seq[int(self.rng.integers(len(seq)))]


def _lower_first(s: str) -> str:
    return s[0].lower() + s[1:] if s and not s.startswith("I ") and not s.startswith("I'") else s


def _bank(code: str) -> dict:
    if code in ph.BANK:
        return ph.BANK[code]
    act = ph.GG_ACTIVITIES[code]
    return {lvl: {v: [t.replace("{act}", act["act"]).replace("{setup}", act["setup"])
                      .replace("{nobs}", act["nobs"]) for t in txts]
                  for v, txts in voices.items()}
            for lvl, voices in ph.GG_LEVELS.items()}


def _less_impaired(code: str, value: str) -> str | None:
    order = ITEM_BY_CODE[code].ordinal
    if value in order and order.index(value) > 0:
        return order[order.index(value) - 1]
    return None


def _question(ctx: _Ctx, code: str) -> str:
    if code in ph.QUESTIONS:
        return ctx.pick(ph.QUESTIONS[code])
    return ph.GG_ACTIVITIES[code]["q"]


def _express_single(ctx: _Ctx, code: str, value: str, style: str) -> tuple[list[dict], str]:
    bank = _bank(code)
    lv = bank[value]
    turns = [{"speaker": "NURSE", "text": _question(ctx, code)}]

    def say(voice: str, val: str = value) -> dict | None:
        opts = bank.get(val, {}).get(voice)
        if not opts:
            return None
        spk = {"P": "PATIENT", "N": "NURSE", "C": ctx.cg_label}[voice]
        if spk is None:
            return None
        return {"speaker": spk, "text": ctx.fmt(ctx.pick(opts))}

    # Fall back gracefully when a voice/style isn't available for this item/patient.
    if style == "caregiver" and (ctx.cg_label is None or "C" not in lv):
        style = "observed" if "N" in lv else "direct"
    if style in ("correction", "temporal"):
        alt = _less_impaired(code, value)
        if alt is None or "P" not in bank.get(alt, {}) or "P" not in lv and style == "temporal":
            style = "observed" if "N" in lv else "direct"
    if style == "direct" and "P" not in lv:
        style = "observed" if "N" in lv else "caregiver"
    if style == "observed" and "N" not in lv:
        style = "direct" if "P" in lv else "caregiver"

    if style in ("direct", "observed", "caregiver"):
        t = say({"direct": "P", "observed": "N", "caregiver": "C"}[style])
        if t is None:  # last resort
            style = "direct"
            t = say("P") or say("N")
        turns.append(t)
    elif style == "correction":
        alt = _less_impaired(code, value)
        turns.append(say("P", alt))
        if ctx.cg_label and "C" in lv:
            fix = say("C")
            fix["text"] = f"{ctx.addr}, that's not really true. " + fix["text"]
            turns.append(fix)
        else:
            turns.append({"speaker": "NURSE", "text": "Let me actually watch you do it, okay?"})
            turns.append({"speaker": "PATIENT", "text": "Okay, okay."})
            obs = say("N") or say("P")
            obs["text"] = "So what I'm seeing today is different. " + obs["text"]
            turns.append(obs)
    elif style == "temporal":
        alt = _less_impaired(code, value)
        before = ctx.fmt(ctx.pick(bank[alt]["P"]))
        now = ctx.fmt(ctx.pick(lv["P"]))
        turns.append({"speaker": "PATIENT",
                      "text": f"Before I went into the hospital it was different. Back then: {before} "
                              f"Not anymore. Now, since I got home, {_lower_first(now)}"})
    return turns, style


def _express_m1033(ctx: _Ctx, marked: list[str]) -> list[dict]:
    turns = [{"speaker": "NURSE", "text": ph.QUESTIONS["M1033"][0]}]
    for r in marked:
        if r == "10":
            continue
        if r == "5":  # option 0 is caregiver voice, option 1 patient voice
            spk = ctx.cg_label or "PATIENT"
            txt = ctx.fmt(ph.M1033_RISKS["5"][0 if ctx.cg_label else 1])
        else:
            spk = "NURSE" if r == "7" else "PATIENT"
            txt = ctx.fmt(ctx.pick(ph.M1033_RISKS[r]))
        turns.append({"speaker": spk, "text": txt})
    for r, denial in ph.M1033_DENIALS.items():
        if r not in marked and (ctx.rng.random() < 0.7 or marked == ["10"]):
            turns.append({"speaker": "PATIENT", "text": denial})
    return turns


def _noise(ctx: _Ctx, text: str, rate: float) -> str:
    if rate <= 0 or ctx.rng.random() > rate:
        return text
    words = text.split()
    pos = int(ctx.rng.integers(0, max(1, len(words))))
    words.insert(pos, ctx.pick(FILLERS) + ",")
    if ctx.rng.random() < 0.4 and len(words) > 3:  # stutter / repeated word
        j = int(ctx.rng.integers(0, len(words)))
        words.insert(j, words[j])
    return " ".join(words)


def make_transcript(p: Patient, tier: str, rng: np.random.Generator, tid: str) -> Transcript:
    cfg = TIERS[tier]
    ctx = _Ctx(p, rng)
    styles_names, styles_p = zip(*cfg["styles"].items())
    expected, styles, blocks = {}, {}, []

    adl_order = ["M1800", "M1810", "M1820", "M1830", "M1840", "M1850", "M1860"] + [c for c, _ in GG_ITEMS]
    rng.shuffle(adl_order)
    order = ["M2020", "M1033"] + adl_order + ["M1400", "J0510"]
    assert sorted(order) == sorted(ITEM_CODES)

    for code in order:
        value = p.answers[code]
        if rng.random() < cfg["omit"]:
            expected[code], styles[code] = None, "omitted"
            continue
        if code == "M1033":
            blocks.append(_express_m1033(ctx, value))
            expected[code], styles[code] = sorted(value, key=int), "history"
            continue
        if code == "J0510":
            style = "observed" if value == "8" else "direct"
        else:
            style = str(rng.choice(styles_names, p=np.array(styles_p) / sum(styles_p)))
        turns, used = _express_single(ctx, code, value, style)
        blocks.append(turns)
        expected[code], styles[code] = value, used

    turns: list[dict] = [{"speaker": "NURSE", "text": f"Hi {p.name}, I'm the nurse from the home health agency. "
                                                      "This is your start of care visit, so I'll be asking a lot of questions today."}]
    talk = list(ph.SMALL_TALK)
    rng.shuffle(talk)
    talk = [t for t in talk if t[0] != "C" or ctx.cg_label][: cfg["small_talk"]]
    distractors = list(ph.OTHER_PERSON_DISTRACTORS) if tier == "hard" else []
    distractors = [d for d in distractors if d[0] != "C" or ctx.cg_label]
    for i, b in enumerate(blocks):
        turns.extend(b)
        if talk and rng.random() < 0.45:
            spk, txt = talk.pop()
            turns.append({"speaker": {"P": "PATIENT", "N": "NURSE", "C": ctx.cg_label}[spk], "text": txt})
        if distractors and rng.random() < 0.15:
            spk, txt = distractors.pop(int(rng.integers(len(distractors))))
            turns.append({"speaker": {"P": "PATIENT", "C": ctx.cg_label}[spk], "text": txt})
    turns.append({"speaker": "NURSE", "text": "Okay, that's everything for today. I'll see you Thursday."})
    for t in turns:
        t["text"] = _noise(ctx, t["text"], cfg["noise"])

    return Transcript(tid, p.patient_id, tier, p.clinical_group, p.timing, p.admission_source,
                      p.comorbidity, turns, expected, styles, p.answers)


def transcript_dict(t: Transcript) -> dict:
    d = asdict(t)
    d["text"] = t.text
    d["synthetic"] = True
    return d


def write_jsonl(rows: list[dict], path) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
