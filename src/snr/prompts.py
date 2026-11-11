"""Prompt templates. One template per slot, identical for real entities, ghosts and the
generic baseline, so the ONLY thing that differs between them is the name.

Descriptors stated in the prompt: profession and birth decade. The name carries its own
origin cue. Never state nationality, sport or field in the prompt (that would leak the slot).
"""
from __future__ import annotations

SLOTS = {
    "citizenship": {"question": "What is {poss} country of citizenship?", "kind": "country name"},
    "sport": {"question": "Which sport is {subj} known for?", "kind": "sport"},
    "field": {"question": "What is {poss} main field of work?", "kind": "field"},
}

DESCRIPTOR = "{Subj} is a {profession} born in the {decade}s."
BASE_TEMPLATE = "{descriptor}\nQuestion: {question}\nAnswer:"
CHAT_TEMPLATE = "{descriptor} {question} Answer with the {kind} only, no other words."

GENERIC = {"Subj": "This person", "subj": "this person", "poss": "this person's"}


def possessive(name: str) -> str:
    return f"{name}'s"


def slot_text(slot: str, name: str | None, profession: str, decade: int, chat: bool) -> str:
    """name=None gives the generic-subject baseline (WikiMem-style 'this person')."""
    if name is None:
        forms = GENERIC
    else:
        forms = {"Subj": name, "subj": name, "poss": possessive(name)}
    spec = SLOTS[slot]
    descriptor = DESCRIPTOR.format(Subj=forms["Subj"], profession=profession, decade=decade)
    question = spec["question"].format(subj=forms["subj"], poss=forms["poss"])
    if chat:
        return CHAT_TEMPLATE.format(descriptor=descriptor, question=question, kind=spec["kind"])
    return BASE_TEMPLATE.format(descriptor=descriptor, question=question)


def candidate_continuation(value: str, chat: bool) -> str:
    """Base prompts end with 'Answer:' so the value follows a space. Chat prompts end with
    the assistant header, so the value starts immediately."""
    return value if chat else " " + value
