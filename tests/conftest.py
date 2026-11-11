import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CORPUS = [
    "Rahul Sharma is a scientist born in the 1960s.",
    "Question: What is Rahul Sharma's country of citizenship?\nAnswer: India",
    "Which sport is Ana Silva known for? Answer with the sport only, no other words.",
    "India Brazil Germany football cricket tennis physics chemistry mathematics",
    "This person is an athlete born in the 1980s. What is this person's main field of work?",
    "Kumar Iyer Santos Muller Schmidt Ferreira Patel Weber Costa Gupta",
]


@pytest.fixture(scope="session")
def tok():
    from snr.tiny import tiny_tokenizer

    return tiny_tokenizer(CORPUS)


@pytest.fixture(scope="session")
def model(tok):
    from snr.tiny import tiny_model

    return tiny_model(tok)


def synthetic_rows():
    """People in three name-origin groups; athletes (sport) and scientists (field)."""
    rows = []
    groups = {
        "India": (["Rahul", "Amit", "Priya", "Sunil", "Anil", "Kavita", "Ravi", "Neha"],
                  ["Sharma", "Gupta", "Patel", "Iyer"], "cricket", "physics"),
        "Brazil": (["Ana", "Joao", "Pedro", "Lucas", "Maria", "Paulo", "Rafael", "Bruna"],
                   ["Silva", "Santos", "Costa", "Ferreira"], "association football", "chemistry"),
        "Germany": (["Hans", "Karl", "Anna", "Lena", "Jonas", "Uwe", "Klaus", "Petra"],
                    ["Muller", "Schmidt", "Weber", "Fischer"], "handball", "mathematics"),
    }
    q = 0
    for country, (given, sur, sport, field) in groups.items():
        for gi, g in enumerate(given):
            for si, s in enumerate(sur):
                if (gi + si) % 2:  # leave half the name combinations unused, so ghosts exist
                    continue
                q += 1
                year = 1960 + (gi % 2) * 5
                # a few atypical people: Brazilian citizen with an Indian surname, etc.
                cit = country
                if country == "India" and gi == 0 and si == 0:
                    cit = "Brazil"
                rows.append({"qid": f"Q{q}", "name": f"{g} {s}", "birth_year": year, "group": "athlete",
                             "citizenship": cit, "slot": "sport", "value": sport if gi != 1 else "tennis",
                             "sitelinks": 1 + (q * 7) % 40})
                q += 1
                rows.append({"qid": f"Q{q}", "name": f"{g} {s}", "birth_year": year + 2, "group": "scientist",
                             "citizenship": cit, "slot": "field", "value": field,
                             "sitelinks": 1 + (q * 5) % 30})
    # a leaky name and a single-token name
    rows.append({"qid": "Q9001", "name": "Pierre France", "birth_year": 1961, "group": "athlete",
                 "citizenship": "France", "slot": "sport", "value": "cricket", "sitelinks": 3})
    rows.append({"qid": "Q9002", "name": "Pele", "birth_year": 1940, "group": "athlete",
                 "citizenship": "Brazil", "slot": "sport", "value": "association football", "sitelinks": 99})
    return rows


def small_cfg():
    return {
        "groups": ["athlete", "scientist"],
        "profession_of_group": {"athlete": "athlete", "scientist": "scientist"},
        "slot_groups": {"citizenship": ["athlete", "scientist"], "sport": ["athlete"], "field": ["scientist"]},
        "top_k_values": {"citizenship": 5, "sport": 5, "field": 5},
        "origin_min_support": 3, "origin_min_purity": 0.5, "seed": 0,
        "selection": {"min_cell_pool": 3, "max_cells_per_slot": 6, "max_cells_per_origin": 2, "items_per_slot": 24},
        "value_surface": {"association football": "football"},
    }
