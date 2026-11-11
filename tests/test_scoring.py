import numpy as np
import torch

from snr.common import build_prompt
from snr.prompts import candidate_continuation, slot_text
from snr.run_g0 import map_answer
from snr.scoring import encode_pair, score_pairs


def manual_score(model, ids, n_p):
    with torch.no_grad():
        lsm = torch.log_softmax(model(input_ids=torch.tensor([ids])).logits[0].float(), -1)
    return float(sum(lsm[t - 1, ids[t]] for t in range(n_p, len(ids))))


def test_scores_match_manual_and_batch_invariant(tok, model):
    prompts = [slot_text("citizenship", "Rahul Sharma", "scientist", 1960, chat=False),
               build_prompt(tok, slot_text("sport", "Ana Silva", "athlete", 1980, chat=True), chat=True)]
    cands = [[candidate_continuation(v, False) for v in ["India", "Brazil", "Germany"]],
             [candidate_continuation(v, True) for v in ["football", "cricket"]]]
    lp1, nt1, ok1 = score_pairs(model, tok, prompts, cands, add_special_tokens=True, batch_size=1, progress=False)
    lp4, nt4, ok4 = score_pairs(model, tok, prompts, cands, add_special_tokens=True, batch_size=4, progress=False)
    for a, b in zip(lp1, lp4):
        assert np.allclose(a, b, atol=1e-4)
    for i, (p, cs) in enumerate(zip(prompts, cands)):
        for k, c in enumerate(cs):
            p_ids, c_ids, _ = encode_pair(tok, p, c, True)
            assert abs(manual_score(model, p_ids + c_ids, len(p_ids)) - lp1[i][k]) < 1e-4
            assert nt1[i][k] == len(c_ids)


def test_boundary_fallback_flagged(tok, model):
    # 'Ind' + 'ia' merges into 'India' in the joint encoding, so the prefix property fails
    p_ids, c_ids, ok = encode_pair(tok, "Answer: Ind", "ia", True)
    full = tok("Answer: India", add_special_tokens=True).input_ids
    if full[:len(p_ids)] != p_ids:
        assert not ok
    lp, nt, bok = score_pairs(model, tok, ["Answer: Ind"], [["ia"]], add_special_tokens=True, progress=False)
    assert bool(bok[0][0]) == ok


def test_prompts_only_differ_by_name():
    a = slot_text("field", "Rahul Sharma", "scientist", 1960, chat=False)
    b = slot_text("field", "Amit Gupta", "scientist", 1960, chat=False)
    g = slot_text("field", None, "scientist", 1960, chat=False)
    assert a.replace("Rahul Sharma", "X") == b.replace("Amit Gupta", "X")
    assert g.startswith("This person is a scientist born in the 1960s.") and "this person's" in g
    for s in ("citizenship", "sport", "field"):
        t = slot_text(s, "Rahul Sharma", "athlete", 1980, chat=True)
        assert "India" not in t and "cricket" not in t  # no slot leakage


def test_map_answer():
    cands = [{"label": "United States of America", "surface": "United States"},
             {"label": "India", "surface": "India"}, {"label": "association football", "surface": "football"}]
    assert map_answer("The United States.", cands) == 0
    assert map_answer("india", cands) == 1
    assert map_answer("Football (soccer)", cands) == 2
    assert map_answer("I am not sure", cands) is None
