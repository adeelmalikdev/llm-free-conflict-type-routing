"""
Paper 2, 8B generator arm: cached, checkpointed runner.

Same design as wire_generator_day3.py (same prompts, same extraction,
same OOF router probabilities, same arms), with three runnability
changes only:

  1. Generation cache: each (example, route) pair is generated once
     and reused across arms. The original script calls the generator
     separately for the learned, oracle, always-single and
     always-multi arms, which repeats identical calls whenever two
     arms pick the same route (the common case). No prompt, parameter
     or extraction logic changes.
  2. Checkpointing: every finished example is appended to
     day3_8b_checkpoint.jsonl immediately, and a restart skips
     examples already in the checkpoint, so an interrupted overnight
     run resumes instead of starting over.
  3. The Llama context is 3072 (the longest prompt in this split is
     about 1,800 tokens) and threads are set explicitly.

Run:
  .venv/bin/python run_8b_cached.py --model-path models/llama-3.1-8b-instruct-Q4_K_M.gguf
  .venv/bin/python run_8b_cached.py --model-path ... --limit 10   # smoke test

Outputs: day3_8b_checkpoint.jsonl (per example), day3_8b_results.json
(arm metrics, written at the end and also refreshable from the
checkpoint with --score-only).
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from train_router_day2 import FEATURE_NAMES, norm_ans, set_features  # noqa: E402
from wire_generator_day3 import (  # noqa: E402
    build_prompt,
    doc_answers,
    evaluate,
    llama_generate,
    oof_probs,
    pick_route,
    true_labels,
)

DATA = os.path.join(HERE, "RAMDocs_test.jsonl")
CKPT = os.path.join(HERE, "day3_8b_checkpoint.jsonl")
RESULTS = os.path.join(HERE, "day3_8b_results.json")


def score_checkpoint(rows, ckpt_path):
    recs = {}
    if os.path.exists(ckpt_path):
        for line in open(ckpt_path):
            rec = json.loads(line)
            recs[rec["id"]] = rec
    done_rows = [rows[i] for i in sorted(recs)]
    arms = {name: {"routes": [], "answers": [], "lat": [], "calls": [], "tok": []}
            for name in ("learned-router", "oracle-router",
                         "always-single", "always-multi")}
    route_correct = 0
    for i in sorted(recs):
        rec = recs[i]
        route_correct += int(rec["route"] == rec["oracle_route"])
        arms["learned-router"]["routes"].append(rec["route"])
        arms["learned-router"]["answers"].append(rec["arms"]["learned-router"])
        arms["learned-router"]["lat"].append(rec["latency_ms"])
        arms["learned-router"]["calls"].append(1)
        arms["learned-router"]["tok"].append(rec["tokens"].get(rec["route"], 0))
        arms["oracle-router"]["routes"].append(rec["oracle_route"])
        arms["oracle-router"]["answers"].append(rec["arms"]["oracle-router"])
        arms["oracle-router"]["lat"].append(0.0)
        arms["oracle-router"]["calls"].append(1)
        arms["oracle-router"]["tok"].append(rec["tokens"].get(rec["oracle_route"], 0))
        arms["always-single"]["routes"].append("SINGLE")
        arms["always-single"]["answers"].append(rec["arms"]["always-single"])
        arms["always-single"]["lat"].append(0.0)
        arms["always-single"]["calls"].append(1)
        arms["always-single"]["tok"].append(rec["tokens"].get("SINGLE", 0))
        arms["always-multi"]["routes"].append("MULTI")
        arms["always-multi"]["answers"].append(rec["arms"]["always-multi"])
        arms["always-multi"]["lat"].append(0.0)
        arms["always-multi"]["calls"].append(1)
        arms["always-multi"]["tok"].append(rec["tokens"].get("MULTI", 0))
    out = {
        "dataset": "RAMDocs test (500 examples)",
        "generator": "llama-cpp",
        "n_done": len(done_rows),
        "route_accuracy_vs_true_labels": (
            round(route_correct / len(done_rows), 4) if done_rows else None),
        "arms": {},
    }
    for name, a in arms.items():
        if done_rows:
            out["arms"][name] = evaluate(done_rows, a["routes"], a["answers"],
                                         a["lat"], a["calls"], a["tok"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(DATA)]
    if args.limit:
        rows = rows[: args.limit]

    if args.score_only:
        out = score_checkpoint(rows, CKPT)
        json.dump(out, open(RESULTS, "w"), indent=2)
        print(json.dumps(out, indent=2))
        return

    if not args.model_path:
        sys.exit("--model-path is required unless --score-only")

    # Router probabilities, identical to Day 3 (OOF, gradient boosting).
    X_rows, feat_ms = [], []
    for r in rows:
        t_f = time.time()
        X_rows.append(set_features(r))
        feat_ms.append((time.time() - t_f) * 1000.0)
    X = np.array(X_rows, dtype=float)
    y_mis = np.array([1 if true_labels(r)[0] else 0 for r in rows])
    y_amb = np.array([1 if true_labels(r)[1] else 0 for r in rows])
    groups = np.arange(len(rows))
    p_mis_all, p_amb_all = oof_probs(X, y_mis, y_amb,
                                     "gradient_boosting", groups)

    done = set()
    if os.path.exists(CKPT):
        for line in open(CKPT):
            done.add(json.loads(line)["id"])
    print(f"checkpoint: {len(done)} examples already done", flush=True)

    from llama_cpp import Llama
    llm = Llama(model_path=args.model_path, n_ctx=3072, n_threads=4,
                verbose=False)
    print("model loaded", flush=True)

    t_start = time.time()
    n_calls = 0
    with open(CKPT, "a") as ck:
        for i, r in enumerate(rows):
            if i in done:
                continue
            p_mis = float(p_mis_all[i])
            p_amb = float(p_amb_all[i])
            n_with = sum(1 for a in doc_answers(r)
                         if a not in ("unknown", ""))
            route = pick_route(p_mis, p_amb, n_with, 0.5, 0.5)
            t_mis, t_amb = true_labels(r)
            oracle_route = pick_route(1.0 if t_mis else 0.0,
                                      1.0 if t_amb else 0.0,
                                      n_with, 0.5, 0.5)
            needed = []
            for rt in (route, oracle_route, "SINGLE", "MULTI"):
                if rt not in needed:
                    needed.append(rt)
            cache = {}
            tok_by_route = {}
            t0 = time.time()
            for rt in needed:
                ans, calls, tok = llama_generate(llm, r, rt)
                cache[rt] = ans
                tok_by_route[rt] = tok
                n_calls += calls
            lat_ms = (time.time() - t0) * 1000.0 + feat_ms[i]
            rec = {
                "id": i,
                "question": r["question"],
                "p_misinfo": round(p_mis, 4),
                "p_ambiguous": round(p_amb, 4),
                "true_misinfo": t_mis,
                "true_ambiguous": t_amb,
                "route": route,
                "oracle_route": oracle_route,
                "arms": {
                    "learned-router": cache[route],
                    "oracle-router": cache[oracle_route],
                    "always-single": cache["SINGLE"],
                    "always-multi": cache["MULTI"],
                },
                "gold_answers": [norm_ans(a) for a in r["gold_answers"]],
                "tokens": tok_by_route,
                "latency_ms": round(lat_ms, 1),
            }
            ck.write(json.dumps(rec) + "\n")
            ck.flush()
            done.add(i)
            if len(done) % 10 == 0 or len(done) == len(rows):
                el = time.time() - t_start
                rate = len(done) / el if el else 0
                eta_h = (len(rows) - len(done)) / rate / 3600 if rate else -1
                print(f"{len(done)}/{len(rows)} done, {n_calls} calls this "
                      f"session, elapsed {el/60:.1f} min, ETA "
                      f"{eta_h:.2f} h", flush=True)

    out = score_checkpoint(rows, CKPT)
    out["model_path"] = args.model_path
    json.dump(out, open(RESULTS, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print("Wrote", RESULTS)


if __name__ == "__main__":
    main()
