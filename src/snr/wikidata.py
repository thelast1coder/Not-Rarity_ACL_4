"""Wikidata queries for real people with slot values (citizenship, sport, field of work).

Two endpoints:
  wdqs   https://query.wikidata.org/sparql           (official; 60 s timeout, heavy queries fail)
  qlever https://qlever.dev/api/wikidata  (fast; recommended for these sizes)

QIDs/PIDs below were written from memory. The builder prints a sanity report; if a group
comes back empty or tiny, look the QID up on wikidata.org, fix configs/g0.yaml, and log
the fix in notes/decisions.md. Wikidata content is data: never execute anything in it.
"""
from __future__ import annotations

import re
import time

YEAR_RE = re.compile(r"^\+?(\d{4})-")

ENDPOINTS = {
    "wdqs": "https://query.wikidata.org/sparql",
    "qlever": "https://qlever.dev/api/wikidata",
}

PREFIXES = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""

# sportspeople: anyone with P641 (sport). Slots: citizenship, sport.
ATHLETE_QUERY = PREFIXES + """
SELECT ?p ?name ?dob ?cit ?citLabel ?val ?valLabel ?sl WHERE {{
  ?p wdt:P31 wd:Q5 ; wdt:P641 ?val ; wdt:P569 ?dob ; wdt:P27 ?cit ; wikibase:sitelinks ?sl .
  ?p rdfs:label ?name . FILTER(LANG(?name) = "en")
  ?cit rdfs:label ?citLabel . FILTER(LANG(?citLabel) = "en")
  ?val rdfs:label ?valLabel . FILTER(LANG(?valLabel) = "en")
  FILTER(YEAR(?dob) >= {y0} && YEAR(?dob) <= {y1})
}} LIMIT {limit}
"""

# scientists: occupation in a fixed list AND a field of work (P101). Slots: citizenship, field.
SCIENTIST_QUERY = PREFIXES + """
SELECT ?p ?name ?dob ?cit ?citLabel ?val ?valLabel ?sl WHERE {{
  VALUES ?occ {{ {occupations} }}
  ?p wdt:P31 wd:Q5 ; wdt:P106 ?occ ; wdt:P101 ?val ; wdt:P569 ?dob ; wdt:P27 ?cit ; wikibase:sitelinks ?sl .
  ?p rdfs:label ?name . FILTER(LANG(?name) = "en")
  ?cit rdfs:label ?citLabel . FILTER(LANG(?citLabel) = "en")
  ?val rdfs:label ?valLabel . FILTER(LANG(?valLabel) = "en")
  FILTER(YEAR(?dob) >= {y0} && YEAR(?dob) <= {y1})
}} LIMIT {limit}
"""


def run_sparql(query: str, endpoint: str = "qlever", timeout: int = 600, retries: int = 3) -> list[dict]:
    import requests

    url = ENDPOINTS.get(endpoint, endpoint)
    headers = {"Accept": "application/sparql-results+json",
               "User-Agent": "ErdosAILab-research/0.1 (academic research; contact via repository)"}
    err = None
    for a in range(retries):
        try:
            r = requests.post(url, data={"query": query}, headers=headers, timeout=timeout)
            r.raise_for_status()
            js = r.json()
            return [{k: v["value"] for k, v in b.items()} for b in js["results"]["bindings"]]
        except Exception as e:  # pragma: no cover - network
            err = e
            time.sleep(10 * (a + 1))
    raise RuntimeError(f"SPARQL failed after {retries} attempts on {url}: {err}")


def qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def fetch_group(group: str, cfg: dict, endpoint: str) -> list[dict]:
    """Returns flat rows: person, name, birth year, citizenship label, slot value label, sitelinks."""
    y0, y1 = cfg["birth_year_range"]
    limit = cfg["limit_per_group"]
    if group == "athlete":
        q = ATHLETE_QUERY.format(y0=y0, y1=y1, limit=limit)
        slot = "sport"
    elif group == "scientist":
        occ = " ".join(f"wd:{x}" for x in cfg["scientist_occupations"])
        q = SCIENTIST_QUERY.format(occupations=occ, y0=y0, y1=y1, limit=limit)
        slot = "field"
    else:
        raise ValueError(group)
    rows = run_sparql(q, endpoint=endpoint)
    if len(rows) >= limit:
        raise RuntimeError(f"{group}: query returned {len(rows)} rows = limit_per_group; the result is truncated "
                           "in arbitrary order (whole countries can vanish). Raise data.limit_per_group.")
    out = []
    for r in rows:
        m = YEAR_RE.match(r.get("dob", ""))
        if not m:
            continue
        year = int(m.group(1))
        out.append({"qid": qid(r["p"]), "name": r["name"], "birth_year": year, "group": group,
                    "citizenship": r["citLabel"], "slot": slot, "value": r["valLabel"],
                    "sitelinks": int(float(r["sl"]))})
    return out
