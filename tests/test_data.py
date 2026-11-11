from conftest import small_cfg, synthetic_rows

from snr.build_entities import aggregate, build, leaky, origin_classes, parse_name
from snr.counts import build_queries
from snr.ghosts import make_ghosts
from snr.infinigram import cnf


def test_parse_name():
    assert parse_name("Rahul Sharma") == ("Rahul", "Sharma")
    assert parse_name("John Smith Jr.") == ("John", "Smith")
    assert parse_name("J. Smith") is None
    assert parse_name("Pele") is None
    assert parse_name("Ivan (footballer)") is None


def test_origin_is_leave_one_out():
    rows = synthetic_rows()
    people = aggregate(rows)
    orig = origin_classes(people, min_support=3, min_purity=0.5)
    # 'Rahul Sharma' is the Brazilian citizen with an Indian surname: his own citizenship must
    # not enter his origin class, which comes from the other Sharmas (all Indian).
    rahul = [p for p in people.values() if p["name"] == "Rahul Sharma"][0]
    assert rahul["citizenship"] == {"Brazil"}
    assert orig[rahul["qid"]] == "India"


def test_origin_requires_support():
    rows = [{"qid": f"Q{i}", "name": f"Name{i} Novak", "birth_year": 1950, "group": "athlete",
             "citizenship": "Czech Republic", "slot": "sport", "value": "tennis", "sitelinks": 1} for i in range(3)]
    people = aggregate(rows)
    orig = origin_classes(people, min_support=3, min_purity=0.5)
    assert all(v is None for v in orig.values())  # each sees only 2 others


def test_leaky():
    assert leaky("Pierre France", {"France"}, {})
    assert not leaky("Rahul Sharma", {"India"}, {})


def test_build_outputs():
    res = build(synthetic_rows(), small_cfg())
    items = res["items"]
    assert items, "no items built"
    assert all(it["truth"] for it in items)
    assert not any(it["name"] == "Pierre France" for it in items)
    for it in items:
        o, prof, dec = it["cell"].split("|")
        assert o == it["origin"] and prof == it["profession"] and int(dec) == it["decade"]
        assert it["cell"] in res["cells"][it["slot"]]
    # W counts include every eligible person in the cell
    for slot, cs in res["cells"].items():
        for c in cs.values():
            assert abs(sum(c["w_counts"].values()) - c["pool_size"]) < 1e-6
    # display surface is used for candidates; label kept for truth
    sport = {c["label"]: c["surface"] for c in res["candidates"]["sport"]}
    assert sport.get("association football") == "football"


def test_ghosts_offline_and_verified():
    pools = {"India": {"given": ["Rahul", "Amit", "Priya", "Sunil"], "surname": ["Sharma", "Gupta", "Patel", "Iyer"]}}
    g = make_ghosts(pools, {"Rahul Sharma"}, n_per_origin=5)
    assert len(g["India"]) == 5 and not any(x["verified"] for x in g["India"])
    assert all(x["name"] != "Rahul Sharma" for x in g["India"])
    seen = {"Amit Gupta": 7}
    g2 = make_ghosts(pools, set(), n_per_origin=4, counter=lambda n: seen.get(n, 0))
    assert all(x["verified"] and x["count"] == 0 for x in g2["India"])
    assert all(x["name"] != "Amit Gupta" for x in g2["India"])


def test_count_queries():
    items = [{"item_id": "citizenship:Q1", "slot": "citizenship", "name": "Rahul Sharma", "truth": ["India"]},
             {"item_id": "sport:Q2", "slot": "sport", "name": "Ana Silva", "truth": ["association football"]}]
    cands = {"citizenship": [{"label": "India", "surface": "India"}],
             "sport": [{"label": "association football", "surface": "football"}]}
    qn, qc = build_queries(items, cands, {"India": ["Indian"]})
    assert qn == ["Rahul Sharma", "Ana Silva"]
    assert qc[0] == "Rahul Sharma AND India OR Indian"
    assert qc[1].startswith("Ana Silva AND football OR association football")
    assert cnf(["a AND b"], ["x"]) == "a b AND x"  # reserved words stripped
