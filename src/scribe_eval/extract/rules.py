"""Keyword baseline extractor.

A deliberately simple, transparent baseline of the kind an analyst might prototype
before reaching for an LLM: locate the part of the conversation about an item with
topic keywords, then map cue phrases to a response.

Caveat (stated in the README too): the cue lists were written by the same author as
the synthetic phrase banks, so this baseline effectively knows the vocabulary of the
test set. Treat its scores as an optimistic ceiling for keyword matching, not as what
keywords would achieve on real speech. It does not handle corrections, time contrasts
or statements about someone else, by design.
"""

from __future__ import annotations

import re

from ..items import ITEM_CODES
from .base import Prediction

TOPIC = {
    "M1800": r"groom|comb|brush(ing)? (my|your|her|his) (hair|teeth)|wash(ing)? (my|your) face|shav",
    "M1810": r"shirt|blouse|sleeve|button|dressed up top|cardigan|top on",
    "M1820": r"pants|slacks|socks|shoes|waist down",
    "M1830": r"shower|tub|bath",
    "M1840": r"toilet|commode|bedpan|bathroom",
    "M1850": r"bed to (the )?chair|out of bed|transfer|lift|bedbound|bedfast|turn (me|her|him)",
    "M1860": r"walk|cane|walker|wheelchair|stairs|chairfast|getting around",
    "M1400": r"breath|winded",
    "J0510": r"pain|sleep",
    "M2020": r"pill|medication|medicine|injection",
    "M1033": r"fell|falls|hospital|emergency|ER visit|pounds|confused|exhausted|medications on your list|pills a day",
    "GG0130A": r"eat|feed",
    "GG0130B": r"teeth|denture|toothpaste",
    "GG0130C": r"clean (yourself|myself|up) after|wipes|after toileting",
    "GG0170C": r"lying down to sitting|sit up on the side|edge of the bed|lying to sitting",
    "GG0170D": r"stand up from|standing up",
    "GG0170E": r"bed to chair|bed-to-chair|bed into my chair",
    "GG0170I": r"ten feet|10 feet|doorway",
}

CUES = {  # ordered: first match wins, most impaired first
    "M1800": [("3", r"does all of it|do all of (her|his) grooming|did all of the grooming|can't do any"),
              ("2", r"help|finish"), ("1", r"set (out|up)|placed|tray|in front of"), ("0", r"myself|by yourself|own|without")],
    "M1810": [("3", r"dress myself at all|completely|entirely"), ("2", r"help|guide|buttons"),
              ("1", r"lay|laid|handed"), ("0", r"myself|by yourself|no help|own")],
    "M1820": [("3", r"at all|completely|everything|couldn't participate"), ("2", r"help|had to"),
              ("1", r"put by|laid out|next to"), ("0", r"myself|by yourself|don't help")],
    "M1830": [("6", r"bed bath|bathed completely"), ("5", r"sponge bath.*(help|rest)|need help with most|helps me with the parts"),
              ("4", r"sink|sponge bath"), ("3", r"whole time|entire|throughout"),
              ("2", r"help me step|wash my back|hand stepping|step over"), ("1", r"chair|bench|grab bar"),
              ("0", r"by myself|alone|by yourself")],
    "M1840": [("4", r"dependent|everything for the toilet"), ("3", r"bedpan|urinal"), ("2", r"commode"),
              ("1", r"walks with me|guiding|steady"), ("0", r"by myself|on (her|his) own|yourself")],
    "M1850": [("5", r"turn (me|her|him)|turned and positioned|can't even turn"), ("4", r"roll over|reposition|stay in bed"),
              ("3", r"lift|can't stand|can't put weight"), ("2", r"pivot|a lot"), ("1", r"walker|little|hand on"),
              ("0", r"myself|alone|on your own")],
    "M1860": [("6", r"in bed all|bedfast"), ("5", r"push (her|him) everywhere|can't push|someone has to push"),
              ("4", r"wheel|push my wheelchair|propel"), ("3", r"next to me|beside you|right next"),
              ("2", r"walker"), ("1", r"cane"), ("0", r"without|no cane|no device")],
    "M1400": [("4", r"at rest|doing nothing"), ("3", r"eating|talking"), ("2", r"dress|bathroom"),
              ("1", r"stairs|far|longer distances"), ("0", r"fine|no shortness|doesn't get")],
    "J0510": [("8", r"wasn't able to answer|unable to answer"), ("4", r"almost constantly|barely sleep"),
              ("3", r"frequently|most nights"), ("2", r"occasionally"), ("1", r"rarely"), ("0", r"no pain")],
    "M2020": [("NA", r"injection|injectable"), ("3", r"gives me every|hand (her|him) every|has to give"),
              ("2", r"remind"), ("1", r"pill ?box|organizer|fills"), ("0", r"myself|own pills|correctly told")],
}
GG_CUES = [("88", r"not going to attempt|isn't safe"), ("01", r"can't .* at all|all of my effort|weren't able"),
           ("02", r"most of the work|most of the effort|help a little"), ("03", r"more than half|helps some"),
           ("04", r"steady|cues|remind"), ("05", r"once my|set things up|set up"),
           ("06", r"completely (on my own|by yourself)")]
M1033_CUES = {"1": r"\bfell\b", "2": r"lost .* pounds", "3": r"hospital twice|hospital (two|2|three)",
              "4": r"emergency room (two|three|2|3)", "5": r"confused|forgetful", "6": r"don't always take|stick to the diet",
              "7": r"(seven|eight|nine|ten) (medications|pills)", "8": r"exhausted", "9": r"throw rugs|dark"}


def _windows(lines: list[str], topic: str, span: int = 3) -> list[str]:
    rx = re.compile(topic, re.I)
    out = []
    for i, ln in enumerate(lines):
        if ln.startswith("NURSE:") and rx.search(ln):
            out.append(" ".join(lines[i: i + span + 1]))
    return out


class KeywordExtractor:
    name = "keyword_baseline"

    def extract(self, transcript: str) -> Prediction:
        lines = transcript.splitlines()
        pred: Prediction = {}
        for code in ITEM_CODES:
            pred[code] = {"value": None, "evidence": None}
            if code == "M1033":
                hits = [r for r, rx in M1033_CUES.items() if re.search(rx, transcript, re.I)]
                asked = re.search(TOPIC["M1033"], transcript, re.I)
                pred[code] = {"value": sorted(hits, key=int) if hits else (["10"] if asked else None),
                              "evidence": None}
                continue
            windows = _windows(lines, TOPIC[code])
            if not windows:
                continue
            text = windows[0]
            cues = GG_CUES if code.startswith("GG") else CUES[code]
            for val, rx in cues:
                if re.search(rx, text, re.I):
                    pred[code] = {"value": val, "evidence": text[:200]}
                    break
        return pred
