"""Paper 2, Day 5 stats: bootstrap CIs and McNemar tests, 19 Nov 2026.

Recomputes the four Day 3 stub-generator arms from RAMDocs_test.jsonl
using the same functions as wire_generator_day3.py, verifies the
recomputed means match day3_results.json exactly, then adds what the
results-and-claims draft listed as missing (item 2):
  - percentile bootstrap 95% CIs (10,000 resamples, seed 42) for
    answer-set EM and F1 per arm, and for paired differences
  - exact McNemar (binomial, two-sided) on EM discordant pairs,
    plus chi2 with continuity correction, for learned vs each other arm

All numbers are on the honest cross-fitted (OOF) learned router.
The stub generator means FILTER and SINGLE return the same answer,
so these intervals describe routing, not generation.
"""
import json
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "/home/hatch/workspace/ai-research-discovery/pilot")
from wire_generator_day3 import (  # noqa: E402
    majority_answer, norm_ans, pick_route, set_f1, stub_generate,
    true_labels,
)

HERE = "/home/hatch/workspace/ai-research-discovery/pilot"
rows = [json.loads(l) for l in open(f"{HERE}/RAMDocs_test.jsonl")]
saved = [json.loads(l) for l in open(f"{HERE}/day3_results.jsonl")]
assert len(rows) == len(saved) == 500

# Recompute arms per example.
arms = {"learned-router": [], "oracle-router": [],
        "always-single": [], "always-multi": []}
for i, r in enumerate(rows):
    gold = [norm_ans(a) for a in r["gold_answers"]]
    # learned: route + answers from saved Day 3 record (OOF routing)
    rec = saved[i]
    assert rec["id"] == i and rec["gold_answers"] == gold
    learned_ans = rec["answers"]
    # verify the stub reproduces the saved learned answers from its route
    check, _ = stub_generate(r, rec["route"])
    assert check == learned_ans, f"row {i}: stub mismatch"
    t_mis, t_amb = true_labels(r)
    n_with = sum(1 for d in r["documents"]
                 if norm_ans(d["answer"]) not in ("unknown", ""))
    oracle_route = pick_route(1.0 if t_mis else 0.0,
                              1.0 if t_amb else 0.0, n_with, 0.5, 0.5)
    oracle_ans, _ = stub_generate(r, oracle_route)
    single_ans, _ = stub_generate(r, "SINGLE")
    multi_ans, _ = stub_generate(r, "MULTI")
    arms["learned-router"].append((learned_ans, gold))
    arms["oracle-router"].append((oracle_ans, gold))
    arms["always-single"].append((single_ans, gold))
    arms["always-multi"].append((multi_ans, gold))

# Per-example EM (0/1) and F1 arrays.
em, f1 = {}, {}
for name, pairs in arms.items():
    em[name] = np.array([1.0 if set(a) == set(g) else 0.0
                         for a, g in pairs])
    f1[name] = np.array([set_f1(a, g) for a, g in pairs])

# Verify against the published Day 3 summary.
published = json.load(open(f"{HERE}/day3_results.json"))["arms"]
for name in arms:
    assert abs(em[name].mean() - published[name]["answer_set_em"]) < 0.0005, name
    assert abs(f1[name].mean() - published[name]["answer_set_f1"]) < 0.0005, name
print("Recomputed means match day3_results.json for all four arms.")

rng = np.random.default_rng(42)
B = 10000
n = 500
idx = rng.integers(0, n, size=(B, n))


def boot_ci(arr):
    boots = arr[idx].mean(axis=1)
    return [round(float(np.percentile(boots, q)), 4) for q in (2.5, 97.5)]


def boot_diff_ci(a, b):
    boots = (a[idx] - b[idx]).mean(axis=1)
    return [round(float(np.percentile(boots, q)), 4) for q in (2.5, 97.5)]


def mcnemar(a, b):
    """Exact binomial McNemar on EM discordants; a,b are 0/1 arrays."""
    both = int(((a == 1) & (b == 1)).sum())
    a_only = int(((a == 1) & (b == 0)).sum())   # learned right, other wrong
    b_only = int(((a == 0) & (b == 1)).sum())   # learned wrong, other right
    neither = int(((a == 0) & (b == 0)).sum())
    discord = a_only + b_only
    from scipy.stats import binomtest
    p_exact = float(binomtest(a_only, discord, 0.5,
                              alternative="two-sided").pvalue) if discord else 1.0
    # chi2 with continuity correction
    chi2 = (abs(a_only - b_only) - 1) ** 2 / discord if discord else 0.0
    from scipy.stats import chi2 as chi2_dist
    p_chi2 = float(chi2_dist.sf(chi2, 1))
    return {"both_correct": both, "learned_only_correct": a_only,
            "other_only_correct": b_only, "both_wrong": neither,
            "discordant": discord, "p_exact_two_sided": round(p_exact, 6),
            "chi2_cc": round(float(chi2), 3),
            "p_chi2_cc": round(p_chi2, 6)}


out = {"n": 500, "method": "percentile bootstrap, 10000 resamples, seed 42; "
                            "McNemar exact binomial two-sided on EM",
       "generator": "stub (routing only; FILTER == SINGLE with stub)",
       "router_probs": "oof cross-fitted",
       "arms": {}, "differences": {}, "mcnemar_em": {}}
for name in arms:
    out["arms"][name] = {
        "em": round(float(em[name].mean()), 4),
        "em_ci95": boot_ci(em[name]),
        "f1": round(float(f1[name].mean()), 4),
        "f1_ci95": boot_ci(f1[name]),
    }
pairs = [("learned-router", "always-multi", "em"),
         ("learned-router", "always-single", "em"),
         ("learned-router", "oracle-router", "em"),
         ("learned-router", "always-multi", "f1"),
         ("learned-router", "always-single", "f1"),
         ("learned-router", "oracle-router", "f1")]
for a, b, metric in pairs:
    arr_a = em[a] if metric == "em" else f1[a]
    arr_b = em[b] if metric == "em" else f1[b]
    key = f"{a}_minus_{b}_{metric}"
    out["differences"][key] = {
        "mean_diff": round(float((arr_a - arr_b).mean()), 4),
        "ci95": boot_diff_ci(arr_a, arr_b),
    }
for other in ("always-multi", "always-single", "oracle-router"):
    out["mcnemar_em"][f"learned-router_vs_{other}"] = mcnemar(
        em["learned-router"], em[other])

json.dump(out, open(f"{HERE}/paper2-stats-2026-11-19.json", "w"), indent=2)
print(json.dumps(out, indent=2))
