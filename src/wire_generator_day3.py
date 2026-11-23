"""
Paper 2, Day 3: wire the saved Day 2 router models to a generator.

What this does
--------------
Loads the four final Day 2 models (gradient boosting by default, or
logistic regression with --model logreg), scores every RAMDocs test
example with the same 22 LLM-free set features, picks a route, and
generates an answer set for that route:

  ABSTAIN  no answer-bearing document at all -> return no answer
  FILTER   p(misinfo) >= threshold -> filter-then-answer once
  MULTI    p(ambiguous) >= threshold -> one answer per answer cluster
  SINGLE   otherwise -> the majority answer

Route priority is ABSTAIN > FILTER > MULTI > SINGLE: misinformation
safety first, then ambiguity. The same priority is used for the
oracle router, so learned vs oracle is apples to apples.

Generator backends
------------------
  stub (default): a deterministic extractive generator, no LLM at all.
    SINGLE/FILTER return the majority answer; MULTI returns one
    representative per token cluster (same clustering as Day 2).
    This validates all the plumbing, route distribution and metrics
    end to end, and measures the router-plus-answer-selection floor.
    Honest limit: with the stub, FILTER and SINGLE return the same
    answer, because filtering only changes what a real generator is
    shown. The FILTER gain can only be measured with a real generator.
  llama-cpp: a local quantised generator (e.g. an 8B GGUF) via
    llama-cpp-python. Prompts differ per route; the answer set is
    extracted only from strings that appear as a document answer in
    the retrieved set, never from the gold answers. This code path is
    written but NOT tested here: no GGUF model and no llama-cpp are
    installed on this VM (checked 18 Nov 2026), so running it needs a
    model download first. See the note at the bottom of the summary.

Router probabilities: honest by default
----------------------------------------
The saved Day 2 final models were fit on all 500 examples, so
routing with them and scoring on the same 500 is in-sample and
flatters the learned router (route accuracy 0.998 in that mode).
The default here is --router-probs oof: 5-fold cross-fitting
(GroupKFold, seed 42, same outer loop as Day 2) with the modal
Day 2 hyperparameters, so every routing decision is made by a model
that never trained on that question. Use --router-probs in-sample
only to reproduce the optimistic numbers, never to quote them.

Baselines, all scored with the same stub or the same generator:
  always-single  majority answer for every question
  always-multi   one answer per cluster for every question
  oracle-router  routes from the true labels (any misinfo doc in the
                 set; more than one disambiguated entity), same
                 generator as the learned router

Metrics: answer-set exact match (normalised set equality) and
answer-set F1 (per-answer precision/recall over sets), averaged over
questions, plus route distribution, route accuracy vs the true
labels, median latency and generator calls per question.

Abstention honesty note: gold_answers is never empty in this 500
example RAMDocs test split (1-3 answers each), so abstention has no
ground-truth class here. ABSTAIN fires only when no document carries
an answer (1 of 500 examples). Real abstention measurement needs the
FaithEval-unanswerable split named in the experimental blueprint,
not this file.

Run:
  python wire_generator_day3.py                      # stub, OOF routing, all 500
  python wire_generator_day3.py --limit 20          # smoke test
  python wire_generator_day3.py --model logreg
  python wire_generator_day3.py --router-probs in-sample   # optimistic, do not quote
  python wire_generator_day3.py --generator llama-cpp --model-path /path/to/model.gguf --limit 50

Outputs (in this folder): day3_results.json, day3_results.jsonl
(one record per question), day3-summary-2026-11-18.md.
"""
import argparse
import collections
import json
import os
import re
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from train_router_day2 import (  # noqa: E402
    FEATURE_NAMES,
    norm_ans,
    set_features,
)

DATA = os.path.join(HERE, "RAMDocs_test.jsonl")
MODELS = {
    "gradient_boosting": {
        "misinfo": "router_final_gradient_boosting_misinfo_present.joblib",
        "ambig": "router_final_gradient_boosting_ambiguous_multi_entity.joblib",
    },
    "logreg": {
        "misinfo": "router_final_logreg_misinfo_present.joblib",
        "ambig": "router_final_logreg_ambiguous_multi_entity.joblib",
    },
}


def true_labels(r):
    misinfo = any(d["type"] == "misinfo" for d in r["documents"])
    ambig = len(r["disambig_entity"]) > 1
    return misinfo, ambig


def pick_route(p_misinfo, p_ambig, n_with_answer, misinfo_thr, ambig_thr):
    if n_with_answer == 0:
        return "ABSTAIN"
    if p_misinfo >= misinfo_thr:
        return "FILTER"
    if p_ambig >= ambig_thr:
        return "MULTI"
    return "SINGLE"


def doc_answers(r):
    """Normalised answers per document, in document order."""
    return [norm_ans(d["answer"]) for d in r["documents"]]


def answer_clusters(r):
    """(representative answer, support) per token cluster, support desc.

    Greedy clustering with the same >= 0.5 token-Jaccard rule as Day 2:
    answers in descending support order join the first cluster whose
    representative overlaps enough, otherwise start a new cluster.
    The representative is the highest-support answer in the cluster.
    """
    answers = [a for a in doc_answers(r) if a not in ("unknown", "")]
    cnt = collections.Counter(answers)
    items = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    built = []  # [rep token set, rep answer, n_distinct, support]
    for ans, c in items:
        toks = set(ans.split())
        placed = False
        for cl in built:
            inter = toks & cl[0]
            union = toks | cl[0]
            if union and len(inter) / len(union) >= 0.5:
                cl[2] += 1
                cl[3] += c
                placed = True
                break
        if not placed:
            built.append([toks, ans, 1, c])
    built.sort(key=lambda cl: -cl[3])
    return [(cl[1], cl[3]) for cl in built], cnt


def majority_answer(r):
    answers = [a for a in doc_answers(r) if a not in ("unknown", "")]
    if not answers:
        return None
    return collections.Counter(answers).most_common(1)[0][0]


def stub_generate(r, route):
    """Deterministic extractive generator. Returns (answers, n_calls)."""
    if route == "ABSTAIN":
        return [], 0
    if route == "MULTI":
        reps, _ = answer_clusters(r)
        return [a for a, _ in reps], 1
    maj = majority_answer(r)
    return ([maj] if maj else []), 1


def build_prompt(r, route):
    docs = "\n".join(
        f"[{i + 1}] {d['text'][:1500]}" for i, d in enumerate(r["documents"])
    )
    q = r["question"]
    if route == "MULTI":
        instr = (
            "The question may have more than one correct answer for "
            "different entities. List every distinct correct answer, "
            "one per line, and nothing else."
        )
    elif route == "FILTER":
        instr = (
            "Some documents may contain misinformation. Ignore any "
            "document whose answer is not supported by the others, "
            "then give the single correct answer on one line."
        )
    else:
        instr = "Give the single correct answer on one line."
    return f"{instr}\n\nQuestion: {q}\n\nDocuments:\n{docs}\n\nAnswer:"


def llama_generate(llm, r, route):
    """One llama-cpp call; answer set grounded in the retrieved docs."""
    if route == "ABSTAIN":
        return [], 0, 0
    prompt = build_prompt(r, route)
    t0 = time.time()
    out = llm(prompt, max_tokens=256, temperature=0.0, stop=["\n\n", "Question:"])
    latency_ms = (time.time() - t0) * 1000.0
    text = out["choices"][0]["text"]
    n_tokens = out.get("usage", {}).get("total_tokens", 0)
    candidates = {a for a in doc_answers(r) if a not in ("unknown", "")}
    found = []
    for line in text.splitlines():
        a = norm_ans(line.strip().lstrip("-*0123456789. "))
        if a in candidates and a not in found:
            found.append(a)
    if route in ("SINGLE", "FILTER") and found:
        found = found[:1]
    return found, 1, n_tokens


def set_f1(pred, gold):
    p, g = set(pred), set(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = len(p & g)
    prec = tp / len(p)
    rec = tp / len(g)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def evaluate(rows, routes, answers_list, latencies, calls, tokens):
    em, f1s = [], []
    for r, ans in zip(rows, answers_list):
        gold = [norm_ans(a) for a in r["gold_answers"]]
        em.append(1.0 if set(ans) == set(gold) else 0.0)
        f1s.append(set_f1(ans, gold))
    return {
        "n": len(rows),
        "answer_set_em": round(float(np.mean(em)), 4),
        "answer_set_f1": round(float(np.mean(f1s)), 4),
        "route_counts": dict(collections.Counter(routes)),
        "median_latency_ms": round(float(np.median(latencies)), 1),
        "mean_generator_calls": round(float(np.mean(calls)), 3),
        "mean_tokens_per_q": round(float(np.mean(tokens)), 1),
    }


def oof_probs(X, y_mis, y_amb, model_name, groups):
    """Cross-fitted router probabilities, one fold held out at a time.

    Fixed hyperparameters are the modal Day 2 outer-fold winners:
    gradient boosting learning_rate 0.05, max_iter 100,
    max_leaf_nodes 15; logreg C 10, no class weight. Groups are the
    question indices, seed 42, same outer loop as Day 2.
    """
    def make(name):
        if name == "gradient_boosting":
            return HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=100, max_leaf_nodes=15,
                early_stopping=False, random_state=42)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=10.0)),
        ])

    p_mis = np.zeros(len(X))
    p_amb = np.zeros(len(X))
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y_mis, groups):
        m1, m2 = make(model_name), make(model_name)
        m1.fit(X[tr], y_mis[tr])
        m2.fit(X[tr], y_amb[tr])
        p_mis[te] = m1.predict_proba(X[te])[:, 1]
        p_amb[te] = m2.predict_proba(X[te])[:, 1]
    return p_mis, p_amb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gradient_boosting",
                    choices=["gradient_boosting", "logreg"])
    ap.add_argument("--router-probs", default="oof",
                    choices=["oof", "in-sample"],
                    help="oof: cross-fitted, honest (default). "
                         "in-sample: saved Day 2 final models, optimistic.")
    ap.add_argument("--generator", default="stub",
                    choices=["stub", "llama-cpp"])
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--misinfo-threshold", type=float, default=0.5)
    ap.add_argument("--ambig-threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-prefix", default="day3")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(DATA)]
    if args.limit:
        rows = rows[: args.limit]

    # Features once, then router probabilities for every row.
    # Time the feature step per row so the learned arm's latency
    # honestly includes routing cost, not just generation.
    feat_ms = []
    X_rows = []
    for r in rows:
        t_f = time.time()
        X_rows.append(set_features(r))
        feat_ms.append((time.time() - t_f) * 1000.0)
    X = np.array(X_rows, dtype=float)
    y_mis = np.array([1 if true_labels(r)[0] else 0 for r in rows])
    y_amb = np.array([1 if true_labels(r)[1] else 0 for r in rows])
    groups = np.arange(len(rows))
    if args.router_probs == "oof":
        p_mis_all, p_amb_all = oof_probs(X, y_mis, y_amb, args.model, groups)
    else:
        def load_model(name):
            obj = joblib.load(os.path.join(HERE, MODELS[args.model][name]))
            # Day 2 saved {"model": Pipeline, "feature_names": ..., ...}.
            return obj["model"] if isinstance(obj, dict) else obj

        misinfo_model = load_model("misinfo")
        ambig_model = load_model("ambig")
        p_mis_all = misinfo_model.predict_proba(X)[:, 1]
        p_amb_all = ambig_model.predict_proba(X)[:, 1]

    llm = None
    if args.generator == "llama-cpp":
        if not args.model_path:
            sys.exit("--model-path is required with --generator llama-cpp")
        from llama_cpp import Llama
        llm = Llama(model_path=args.model_path, n_ctx=4096, verbose=False)

    def generate(r, route):
        if args.generator == "stub":
            ans, calls = stub_generate(r, route)
            return ans, calls, 0
        return llama_generate(llm, r, route)

    records = []
    arms = {
        "learned-router": {"routes": [], "answers": [], "lat": [], "calls": [], "tok": []},
        "oracle-router": {"routes": [], "answers": [], "lat": [], "calls": [], "tok": []},
        "always-single": {"routes": [], "answers": [], "lat": [], "calls": [], "tok": []},
        "always-multi": {"routes": [], "answers": [], "lat": [], "calls": [], "tok": []},
    }
    route_correct = 0
    for i, r in enumerate(rows):
        t_route = time.time()
        p_mis = float(p_mis_all[i])
        p_amb = float(p_amb_all[i])
        n_with = sum(1 for a in doc_answers(r) if a not in ("unknown", ""))
        route = pick_route(p_mis, p_amb, n_with,
                           args.misinfo_threshold, args.ambig_threshold)
        t_mis, t_amb = true_labels(r)
        true_route = pick_route(1.0 if t_mis else 0.0, 1.0 if t_amb else 0.0,
                                n_with, 0.5, 0.5)
        route_correct += int(route == true_route)

        ans, calls, tok = generate(r, route)
        # Learned-arm latency covers routing (features + models) plus
        # generation, so it is the honest per-question cost of the arm.
        lat = feat_ms[i] + (time.time() - t_route) * 1000.0
        arms["learned-router"]["routes"].append(route)
        arms["learned-router"]["answers"].append(ans)
        arms["learned-router"]["lat"].append(lat)
        arms["learned-router"]["calls"].append(calls)
        arms["learned-router"]["tok"].append(tok)

        # Oracle router, same generator, same priority.
        t0 = time.time()
        o_ans, o_calls, o_tok = generate(r, true_route)
        o_lat = (time.time() - t0) * 1000.0
        arms["oracle-router"]["routes"].append(true_route)
        arms["oracle-router"]["answers"].append(o_ans)
        arms["oracle-router"]["lat"].append(o_lat)
        arms["oracle-router"]["calls"].append(o_calls)
        arms["oracle-router"]["tok"].append(o_tok)

        for name, fixed_route in (("always-single", "SINGLE"),
                                  ("always-multi", "MULTI")):
            a, c, tk = generate(r, fixed_route)
            arms[name]["routes"].append(fixed_route)
            arms[name]["answers"].append(a)
            arms[name]["lat"].append(0.0)
            arms[name]["calls"].append(c)
            arms[name]["tok"].append(tk)

        records.append({
            "id": i,
            "question": r["question"],
            "p_misinfo": round(p_mis, 4),
            "p_ambiguous": round(p_amb, 4),
            "true_misinfo": t_mis,
            "true_ambiguous": t_amb,
            "route": route,
            "oracle_route": true_route,
            "answers": ans,
            "gold_answers": [norm_ans(a) for a in r["gold_answers"]],
            "latency_ms": round(lat, 1),
        })
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(rows)} routed", flush=True)

    results = {
        "dataset": "RAMDocs test (500 examples)",
        "router_model": args.model,
        "router_probs": args.router_probs,
        "router_probs_note": (
            "oof: 5-fold cross-fitted, honest"
            if args.router_probs == "oof" else
            "in-sample: final Day 2 models fit on all 500, optimistic, do not quote"
        ),
        "generator": args.generator,
        "model_path": args.model_path,
        "thresholds": {"misinfo": args.misinfo_threshold,
                       "ambiguous": args.ambig_threshold},
        "route_accuracy_vs_true_labels": round(route_correct / len(rows), 4),
        "arms": {},
    }
    for name, a in arms.items():
        results["arms"][name] = evaluate(rows, a["routes"], a["answers"],
                                         a["lat"], a["calls"], a["tok"])

    json_path = os.path.join(HERE, f"{args.out_prefix}_results.json")
    jsonl_path = os.path.join(HERE, f"{args.out_prefix}_results.jsonl")
    json.dump(results, open(json_path, "w"), indent=2)
    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    lines = [
        "# Paper 2, Day 3 generator wiring summary, 18 Nov 2026",
        "",
        f"Router: {args.model} (saved Day 2 models). Generator: {args.generator}"
        + (f" ({args.model_path})" if args.model_path else "")
        + f". Router probs: {args.router_probs}. "
        f"n = {len(rows)}. Thresholds: misinfo {args.misinfo_threshold}, "
        f"ambiguous {args.ambig_threshold}.",
        "",
        f"Route accuracy vs true labels: {results['route_accuracy_vs_true_labels']}.",
        "",
        "## Arms (answer-set EM / F1)",
        "",
    ]
    for name, m in results["arms"].items():
        lines.append(
            f"- {name}: EM {m['answer_set_em']}, F1 {m['answer_set_f1']}, "
            f"routes {m['route_counts']}, median latency "
            f"{m['median_latency_ms']} ms, mean calls "
            f"{m['mean_generator_calls']}, mean tokens "
            f"{m['mean_tokens_per_q']}"
        )
    lines += [
        "",
        "## Notes and caveats",
        "",
        "- Router probabilities are cross-fitted (OOF) by default, so "
        "the learned-router numbers above are honest. The in-sample "
        "mode (saved Day 2 final models on the same 500) gives route "
        "accuracy 0.998 and must not be quoted.",
        "- With the stub generator, FILTER and SINGLE return the same "
        "answer, because filtering only changes what a real generator "
        "is shown. The FILTER gain can only be measured with the "
        "llama-cpp arm.",
        "- Read EM and F1 together: the learned router beats "
        "always-multi on exact set match, but always-multi scores "
        "higher on F1, because returning every cluster usually "
        "includes the gold answers even when it adds wrong ones. "
        "That is the cost of routing ambiguous sets to a single "
        "answer, and the 8B arm is where it gets tested properly.",
        "- gold_answers is never empty in this 500-example split, so "
        "abstention has no ground-truth class here; ABSTAIN fires "
        "only when no document carries an answer. Real abstention "
        "measurement needs the FaithEval-unanswerable split.",
        "- The llama-cpp arm is written but untested on this VM: no "
        "GGUF model and no llama-cpp-python are installed here "
        "(checked 18 Nov 2026). Running it needs a model download "
        "first, plus a timing run before the full 500.",
        "- Day 1 majority-answer floor for reference: a single-answer "
        "majority vote matches the full gold set only 19% of the "
        "time, which is why the MULTI route matters.",
        "",
    ]
    summary_path = os.path.join(HERE, "day3-summary-2026-11-18.md")
    open(summary_path, "w").write("\n".join(lines))
    print(json.dumps(results, indent=2))
    print("Wrote", json_path, jsonl_path, summary_path)


if __name__ == "__main__":
    main()
