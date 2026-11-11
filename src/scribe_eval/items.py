"""OASIS-E2 item catalog used by the harness.

Response wording follows the CMS OASIS-E2 Guidance Manual (effective 04/01/2026).
Some long responses are shortened slightly; the response codes and their meaning are
unchanged. The harness covers 18 items chosen because they drive PDGM payment
(M1800-M1860, M1033), feed HHVBP / Care Compare quality measures (M1810-M1830,
GG functional items, M1400, M2020), or are classic QA consistency checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Section GG uses one shared 6-level performance scale plus "activity not attempted" codes.
GG_SCALE: dict[str, str] = {
    "06": "Independent - patient completes the activity by themself with no assistance from a helper",
    "05": "Setup or clean-up assistance - helper sets up or cleans up; patient completes activity",
    "04": "Supervision or touching assistance - helper provides verbal cues and/or touching/steadying "
          "and/or contact guard assistance as patient completes activity",
    "03": "Partial/moderate assistance - helper does LESS THAN HALF the effort",
    "02": "Substantial/maximal assistance - helper does MORE THAN HALF the effort",
    "01": "Dependent - helper does ALL of the effort, or two or more helpers are required",
    "07": "Patient refused",
    "09": "Not applicable - not attempted and the patient did not perform this activity prior to the current illness",
    "10": "Not attempted due to environmental limitations",
    "88": "Not attempted due to medical condition or safety concerns",
}
GG_PERFORMANCE_CODES = ["01", "02", "03", "04", "05", "06"]  # ordinal, 01 = most dependent


@dataclass(frozen=True)
class Item:
    code: str
    name: str
    section: str
    responses: dict[str, str]
    ordinal: list[str] = field(default_factory=list)  # codes in increasing-impairment order
    kind: str = "single"  # "single" or "checklist"
    pdgm: bool = False

    @property
    def valid_codes(self) -> list[str]:
        return list(self.responses)


ITEMS: list[Item] = [
    Item(
        "M1800", "Grooming", "G",
        {
            "0": "Able to groom self unaided, with or without the use of assistive devices or adapted methods",
            "1": "Grooming utensils must be placed within reach before able to complete grooming activities",
            "2": "Someone must assist the patient to groom self",
            "3": "Patient depends entirely upon someone else for grooming needs",
        },
        ordinal=["0", "1", "2", "3"], pdgm=True,
    ),
    Item(
        "M1810", "Current Ability to Dress Upper Body", "G",
        {
            "0": "Able to get clothes out of closets and drawers, put them on and remove them from the upper body without assistance",
            "1": "Able to dress upper body without assistance if clothing is laid out or handed to the patient",
            "2": "Someone must help the patient put on upper body clothing",
            "3": "Patient depends entirely upon another person to dress the upper body",
        },
        ordinal=["0", "1", "2", "3"], pdgm=True,
    ),
    Item(
        "M1820", "Current Ability to Dress Lower Body", "G",
        {
            "0": "Able to obtain, put on, and remove clothing and shoes without assistance",
            "1": "Able to dress lower body without assistance if clothing and shoes are laid out or handed to the patient",
            "2": "Someone must help the patient put on undergarments, slacks, socks or nylons, and shoes",
            "3": "Patient depends entirely upon another person to dress lower body",
        },
        ordinal=["0", "1", "2", "3"], pdgm=True,
    ),
    Item(
        "M1830", "Bathing", "G",
        {
            "0": "Able to bathe self in shower or tub independently, including getting in and out of tub/shower",
            "1": "With the use of devices, is able to bathe self in shower or tub independently, including getting in and out of the tub/shower",
            "2": "Able to bathe in shower or tub with the intermittent assistance of another person (supervision/reminders, "
                 "getting in and out of the shower or tub, or washing difficult to reach areas)",
            "3": "Able to participate in bathing self in shower or tub, but requires presence of another person throughout the bath for assistance or supervision",
            "4": "Unable to use the shower or tub, but able to bathe self independently with or without the use of devices at the sink, in chair, or on commode",
            "5": "Unable to use the shower or tub, but able to participate in bathing self in bed, at the sink, in bedside chair, or on commode, with the assistance or supervision of another person",
            "6": "Unable to participate effectively in bathing and is bathed totally by another person",
        },
        ordinal=["0", "1", "2", "3", "4", "5", "6"], pdgm=True,
    ),
    Item(
        "M1840", "Toilet Transferring", "G",
        {
            "0": "Able to get to and from the toilet and transfer independently with or without a device",
            "1": "When reminded, assisted, or supervised by another person, able to get to and from the toilet and transfer",
            "2": "Unable to get to and from the toilet but is able to use a bedside commode (with or without assistance)",
            "3": "Unable to get to and from the toilet or bedside commode but is able to use a bedpan/urinal independently",
            "4": "Is totally dependent in toileting",
        },
        ordinal=["0", "1", "2", "3", "4"], pdgm=True,
    ),
    Item(
        "M1850", "Transferring", "G",
        {
            "0": "Able to independently transfer",
            "1": "Able to transfer with minimal human assistance or with use of an assistive device",
            "2": "Able to bear weight and pivot during the transfer process but unable to transfer self",
            "3": "Unable to transfer self and is unable to bear weight or pivot when transferred by another person",
            "4": "Bedfast, unable to transfer but is able to turn and position self in bed",
            "5": "Bedfast, unable to transfer and is unable to turn and position self",
        },
        ordinal=["0", "1", "2", "3", "4", "5"], pdgm=True,
    ),
    Item(
        "M1860", "Ambulation/Locomotion", "G",
        {
            "0": "Able to independently walk on even and uneven surfaces and negotiate stairs with or without railings (no human assistance or assistive device)",
            "1": "With the use of a one-handed device (for example, cane, single crutch, hemi-walker), able to independently walk on even and uneven surfaces and negotiate stairs with or without railings",
            "2": "Requires use of a two-handed device (for example, walker or crutches) to walk alone on a level surface and/or requires human supervision or assistance to negotiate stairs or steps or uneven surfaces",
            "3": "Able to walk only with the supervision or assistance of another person at all times",
            "4": "Chairfast, unable to ambulate but is able to wheel self independently",
            "5": "Chairfast, unable to ambulate and is unable to wheel self",
            "6": "Bedfast, unable to ambulate or be up in a chair",
        },
        ordinal=["0", "1", "2", "3", "4", "5", "6"], pdgm=True,
    ),
    Item(
        "M1033", "Risk of Hospitalization (check all that apply)", "M",
        {
            "1": "History of falls (2 or more falls - or any fall with an injury - in the past 12 months)",
            "2": "Unintentional weight loss of a total of 10 pounds or more in the past 12 months",
            "3": "Multiple hospitalizations (2 or more) in the past 6 months",
            "4": "Multiple emergency department visits (2 or more) in the past 6 months",
            "5": "Decline in mental, emotional, or behavioral status in the past 3 months",
            "6": "Reported or observed history of difficulty complying with any medical instructions (for example, medications, diet, exercise) in the past 3 months",
            "7": "Currently taking 5 or more medications",
            "8": "Currently reports exhaustion",
            "9": "Other risk(s) not listed in 1-8",
            "10": "None of the above",
        },
        kind="checklist", pdgm=True,
    ),
    Item(
        "M1400", "When is the patient dyspneic or noticeably Short of Breath?", "M",
        {
            "0": "Patient is not short of breath",
            "1": "When walking more than 20 feet, climbing stairs",
            "2": "With moderate exertion (for example, while dressing, using commode or bedpan, walking distances less than 20 feet)",
            "3": "With minimal exertion (for example, while eating, talking, or performing other ADLs) or with agitation",
            "4": "At rest (during day or night)",
        },
        ordinal=["0", "1", "2", "3", "4"],
    ),
    Item(
        "J0510", "Pain Effect on Sleep (past 5 days)", "J",
        {
            "0": "Does not apply - I have not had any pain or hurting in the past 5 days",
            "1": "Rarely or not at all",
            "2": "Occasionally",
            "3": "Frequently",
            "4": "Almost constantly",
            "8": "Unable to answer",
        },
        ordinal=["0", "1", "2", "3", "4"],
    ),
    Item(
        "M2020", "Management of Oral Medications", "N",
        {
            "0": "Able to independently take the correct oral medication(s) and proper dosage(s) at the correct times",
            "1": "Able to take medication(s) at the correct times if individual dosages are prepared in advance by another person, OR another person develops a drug diary or chart",
            "2": "Able to take medication(s) at the correct times if given reminders by another person at the appropriate times",
            "3": "Unable to take medication unless administered by another person",
            "NA": "No oral medications prescribed",
        },
        ordinal=["0", "1", "2", "3"],
    ),
]

GG_ITEMS = [
    ("GG0130A", "Eating"),
    ("GG0130B", "Oral hygiene"),
    ("GG0130C", "Toileting hygiene"),
    ("GG0170C", "Lying to sitting on side of bed"),
    ("GG0170D", "Sit to stand"),
    ("GG0170E", "Chair/bed-to-chair transfer"),
    ("GG0170I", "Walk 10 feet"),
]
for _code, _name in GG_ITEMS:
    ITEMS.append(
        Item(_code, f"{_name} (SOC/ROC performance)", "GG", dict(GG_SCALE),
             ordinal=["06", "05", "04", "03", "02", "01"])
    )

ITEM_BY_CODE: dict[str, Item] = {i.code: i for i in ITEMS}
ITEM_CODES: list[str] = [i.code for i in ITEMS]
PDGM_ITEMS = [i.code for i in ITEMS if i.pdgm]


def ordinal_rank(code: str, value: str) -> int | None:
    """Position of a response on the item's impairment scale (0 = least impaired)."""
    item = ITEM_BY_CODE[code]
    if value in item.ordinal:
        return item.ordinal.index(value)
    return None


def catalog_as_dict() -> list[dict]:
    return [
        {"code": i.code, "name": i.name, "section": i.section, "kind": i.kind,
         "pdgm": i.pdgm, "responses": i.responses, "ordinal": i.ordinal}
        for i in ITEMS
    ]
