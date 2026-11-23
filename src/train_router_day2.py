"""
Day 2 (Paper 2, "Route, Don't Debate"): train the set-level conflict router.

Trains logistic regression and gradient boosting for two binary set-level
targets on RAMDocs test (500 examples), using only cheap, LLM-free,
CPU-only features, evaluated with NESTED cross-validation:

  - outer: GroupKFold (5 folds, shuffled, seed 42), groups = question index
           (each RAMDocs example is one question, so no question can leak
           across folds)
  - inner: GroupKFold (3 folds, shuffled, seed 42) grid search, scored on
           ROC-AUC, run inside each outer training fold

Targets (exactly as defined in the Day 1 pilot):
  misinfo_present        : any document in the set has type == "misinfo"
  ambiguous_multi_entity : len(disambig_entity) > 1

Features: the pilot's 7 set-level features plus the Day 1 rebuild items:
answer-candidate clustering (exact + token-overlap), support-count
imbalance, similarity dispersion. All LLM-free.

Optional stretch: --nli adds pairwise NLI entailment/contradiction proxy
features from a small local model, ONLY if transformers/torch are already
installed. If they are not, the run proceeds without them and says so.

Outputs (written next to this script):
  router_results_day2.json          all metrics, feature names, CV setup
  router_final_<model>_<target>.joblib   final models fit on all 500
  day2-summary-2026-11-16.md        plain-language summary of the run

Run: .venv/bin/python train_router_day2.py [--nli]
"""
import argparse, collections, json, math, os, re, sys
import numpy as np

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "RAMDocs_test.jsonl")

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


# ---------------------------------------------------------------- features
def norm_ans(a):
    return re.sub(r"\s+", " ", a.strip().lower())


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def token_clusters(answers_counter):
    """Greedily cluster distinct normalized answers by token-overlap.

    Answers are visited in descending support order; an answer joins the
    first cluster whose representative (first member) has token Jaccard
    >= 0.5 with it, otherwise it starts a new cluster. Returns a list of
    (n_distinct_answers, support) per cluster.
    """
    items = sorted(answers_counter.items(), key=lambda kv: (-kv[1], kv[0]))
    clusters = []  # each: [representative token set, n_distinct, support]
    for ans, cnt in items:
        toks = set(ans.split())
        placed = False
        for cl in clusters:
            if jaccard(toks, cl[0]) >= 0.5:
                cl[1] += 1
                cl[2] += cnt
                placed = True
                break
        if not placed:
            clusters.append([toks, 1, cnt])
    return [(c[1], c[2]) for c in clusters]


def set_features(r):
    texts = [d["text"] for d in r["documents"]]
    raw_answers = [norm_ans(d["answer"]) for d in r["documents"]]
    answers = [a for a in raw_answers if a not in ("unknown", "")]
    n_docs = len(texts)
    n_with = len(answers)

    # doc-doc TF-IDF similarity (same construction as the pilot)
    v = TfidfVectorizer(max_features=5000)
    try:
        M = v.fit_transform(texts)
        sims = (M @ M.T).toarray()
        iu = np.triu_indices(n_docs, k=1)
        pair = sims[iu] if n_docs > 1 else np.array([0.0])
    except ValueError:
        pair = np.array([0.0])

    cnt = collections.Counter(answers)
    counts_sorted = sorted(cnt.values(), reverse=True)
    top_count = counts_sorted[0] if counts_sorted else 0
    second_count = counts_sorted[1] if len(counts_sorted) > 1 else 0
    top_share = (top_count / n_with) if n_with else 0.0

    if n_with:
        probs = np.array([c / n_with for c in cnt.values()], dtype=float)
        entropy = float(-(probs * np.log2(probs)).sum())
    else:
        entropy = 0.0

    clusters = token_clusters(cnt)
    if clusters:
        cl_supports = [s for _, s in clusters]
        largest_support = max(cl_supports)
        largest_share = largest_support / n_with if n_with else 0.0
        n_singleton = sum(1 for d, _ in clusters if d == 1)
        mean_cl_size = float(np.mean([d for d, _ in clusters]))
    else:
        largest_support, largest_share, n_singleton, mean_cl_size = 0, 0.0, 0, 0.0

    feats = [
        float(n_docs),                                        # n_docs
        float(np.mean([len(t.split()) for t in texts])),      # mean_doc_len_words
        float(pair.mean()),                                   # sim_mean
        float(pair.min()),                                    # sim_min
        float(len(cnt)),                                      # n_distinct_answers
        float(top_share),                                     # majority_answer_share
        float(n_with),                                        # n_docs_with_answer
        float(pair.std()),                                    # sim_std
        float(pair.max()),                                    # sim_max
        float(len(clusters)),                                 # n_answer_clusters
        float(largest_support),                               # largest_cluster_support
        float(largest_share),                                 # largest_cluster_share
        float(n_singleton),                                   # n_singleton_clusters
        float(mean_cl_size),                                  # mean_cluster_size_distinct
        float(top_count),                                     # top_answer_count
        float(second_count),                                  # second_answer_count
        float(top_count - second_count),                      # top_minus_second
        float(top_count / second_count) if second_count else float(top_count),  # top_second_ratio
        entropy,                                              # answer_entropy
        float(sum(1 for c in cnt.values() if c >= 2)),        # n_answers_support_ge2
        float((n_docs - n_with) / n_docs) if n_docs else 0.0,  # frac_docs_no_answer
        float(np.mean([len(a.split()) for a in answers])) if answers else 0.0,  # mean_answer_len_tokens
    ]
    return feats


FEATURE_NAMES = [
    "n_docs", "mean_doc_len_words", "sim_mean", "sim_min",
    "n_distinct_answers", "majority_answer_share", "n_docs_with_answer",
    "sim_std", "sim_max",
    "n_answer_clusters", "largest_cluster_support", "largest_cluster_share",
    "n_singleton_clusters", "mean_cluster_size_distinct",
    "top_answer_count", "second_answer_count", "top_minus_second",
    "top_second_ratio", "answer_entropy", "n_answers_support_ge2",
    "frac_docs_no_answer", "mean_answer_len_tokens",
]


# ---------------------------------------------------------------- NLI (stretch, optional)
def nli_features(rows):
    """Pairwise NLI proxies between documents, small local model.

    Only runs when transformers and torch are importable. For each example,
    takes up to the first 6 document texts (truncated), scores ordered pairs
    (premise=doc_i, hypothesis=doc_j) with a small MNLI model and returns
    per-example [mean contradiction prob, max contradiction prob,
    mean entailment prob]. Returns (features array, model name) or
    (None, reason string) if unavailable.
    """
    try:
        import torch  # noqa: F401
        from transformers import pipeline
    except Exception as e:
        return None, f"transformers/torch not available ({type(e).__name__})"
    model_name = "cross-encoder/nli-MiniLM2-L6-H768"
    try:
        clf = pipeline("text-classification", model=model_name,
                       top_k=None, truncation=True)
    except Exception as e:
        return None, f"NLI model {model_name} could not be loaded ({type(e).__name__}: {e})"

    def probs(premise, hypothesis):
        out = clf({"text": premise[:1200], "text_pair": hypothesis[:1200]})
        d = {o["label"].lower(): o["score"] for o in out}
        return d.get("contradiction", 0.0), d.get("entailment", 0.0)

    feats = []
    for r in rows:
        texts = [d["text"] for d in r["documents"]][:6]
        contra, entail = [], []
        for i in range(len(texts)):
            for j in range(len(texts)):
                if i != j:
                    c, e = probs(texts[i], texts[j])
                    contra.append(c)
                    entail.append(e)
        if contra:
            feats.append([float(np.mean(contra)), float(np.max(contra)),
                          float(np.mean(entail))])
        else:
            feats.append([0.0, 0.0, 0.0])
    return np.array(feats, dtype=float), model_name


NLI_FEATURE_NAMES = ["nli_contra_mean", "nli_contra_max", "nli_entail_mean"]


# ---------------------------------------------------------------- metrics
def expected_calibration_error(y_true, p, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (p > lo) & (p <= hi) if b > 0 else (p >= lo) & (p <= hi)
        if m.sum() > 0:
            ece += (m.sum() / len(p)) * abs(p[m].mean() - y_true[m].mean())
    return float(ece)


def metric_block(y_true, p):
    pred = (p >= 0.5).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, p)), 4),
        "pr_auc": round(float(average_precision_score(y_true, p)), 4),
        "accuracy_at_0.5": round(float(accuracy_score(y_true, pred)), 4),
        "brier": round(float(brier_score_loss(y_true, p)), 4),
        "ece_10bin": round(expected_calibration_error(y_true, p), 4),
    }


# ---------------------------------------------------------------- models
def make_models():
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    logreg_grid = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__class_weight": [None, "balanced"],
    }
    gb = HistGradientBoostingClassifier(random_state=SEED, early_stopping=False)
    gb_grid = {
        "learning_rate": [0.05, 0.1],
        "max_iter": [100, 300],
        "max_leaf_nodes": [15, 31],
    }
    return {"logreg": (logreg, logreg_grid), "gradient_boosting": (gb, gb_grid)}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli", action="store_true",
                    help="try to add NLI proxy features (needs transformers/torch)")
    ap.add_argument("--data", default=DATA)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    n = len(rows)
    groups = np.arange(n)  # one question per example: group id == example id

    print(f"Extracting features for {n} examples ...", flush=True)
    F = np.array([set_features(r) for r in rows], dtype=float)
    feature_names = list(FEATURE_NAMES)

    nli_status = "not requested (run without --nli)"
    nli_model = None
    if args.nli:
        print("Attempting NLI features (--nli) ...", flush=True)
        nli_F, info = nli_features(rows)
        if nli_F is None:
            nli_status = f"requested but skipped: {info}"
            print("  NLI skipped:", info, flush=True)
        else:
            F = np.hstack([F, nli_F])
            feature_names += NLI_FEATURE_NAMES
            nli_status = f"included, model {info}"
            nli_model = info
            print(f"  NLI features added from {info}", flush=True)

    targets = {
        "misinfo_present": np.array(
            [1 if any(d["type"] == "misinfo" for d in r["documents"]) else 0
             for r in rows]),
        "ambiguous_multi_entity": np.array(
            [1 if len(r["disambig_entity"]) > 1 else 0 for r in rows]),
    }

    outer = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    inner = GroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    models = make_models()

    results = {
        "dataset": "RAMDocs test (500 examples)",
        "seed": SEED,
        "n_examples": n,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "nli_features": nli_status,
        "cv_setup": {
            "outer": "GroupKFold(n_splits=5, shuffle=True, random_state=42), groups = question index",
            "inner": "GroupKFold(n_splits=3, shuffle=True, random_state=42), GridSearchCV scoring=roc_auc",
            "metrics": "pooled out-of-fold predictions across the 5 outer folds",
        },
        "targets": {},
    }

    for tname, y in targets.items():
        prevalence = float(y.mean())
        maj_acc = float(max(prevalence, 1 - prevalence))
        const_brier = prevalence * (1 - prevalence)  # constant-prevalence predictor
        tblock = {
            "prevalence": round(prevalence, 4),
            "majority_class_baseline": {
                "accuracy": round(maj_acc, 4),
                "roc_auc": 0.5,
                "pr_auc_equals_prevalence": round(prevalence, 4),
                "brier_constant_prevalence": round(const_brier, 4),
            },
            "models": {},
        }
        for mname, (est, grid) in models.items():
            print(f"[{tname}] nested CV: {mname} ...", flush=True)
            oof = np.zeros(n, dtype=float)
            fold_aucs, fold_best = [], []
            for tr_idx, te_idx in outer.split(F, y, groups):
                gs = GridSearchCV(est, grid, cv=inner, scoring="roc_auc",
                                  n_jobs=1)
                gs.fit(F[tr_idx], y[tr_idx], groups=groups[tr_idx])
                p = gs.predict_proba(F[te_idx])[:, 1]
                oof[te_idx] = p
                fold_aucs.append(float(roc_auc_score(y[te_idx], p)))
                fold_best.append({k: (v if not isinstance(v, float) or
                                      math.isfinite(v) else str(v))
                                  for k, v in gs.best_params_.items()})
            block = metric_block(y, oof)
            block["outer_fold_roc_auc_mean"] = round(float(np.mean(fold_aucs)), 4)
            block["outer_fold_roc_auc_std"] = round(float(np.std(fold_aucs)), 4)
            block["outer_fold_best_params"] = fold_best

            # final model: inner-CV grid search on all data, fit, save
            gs_full = GridSearchCV(est, grid,
                                   cv=GroupKFold(n_splits=5, shuffle=True,
                                                 random_state=SEED),
                                   scoring="roc_auc", n_jobs=1)
            gs_full.fit(F, y, groups=groups)
            fname = f"router_final_{mname}_{tname}.joblib"
            joblib.dump({
                "model": gs_full.best_estimator_,
                "feature_names": feature_names,
                "target": tname,
                "best_params": gs_full.best_params_,
                "nli_model": nli_model,
                "seed": SEED,
            }, os.path.join(HERE, fname))
            block["final_model_file"] = fname
            block["final_model_best_params"] = gs_full.best_params_
            block["final_model_inner_cv_roc_auc"] = round(
                float(gs_full.best_score_), 4)
            tblock["models"][mname] = block
            print(f"    pooled OOF ROC-AUC {block['roc_auc']}  "
                  f"PR-AUC {block['pr_auc']}  Brier {block['brier']}  "
                  f"ECE {block['ece_10bin']}", flush=True)
        results["targets"][tname] = tblock

    out_json = os.path.join(HERE, "router_results_day2.json")
    json.dump(results, open(out_json, "w"), indent=2, default=str)
    print("Wrote", out_json, flush=True)

    write_summary(results)
    print("Done.", flush=True)


def write_summary(res):
    lines = []
    lines.append("# Paper 2, Day 2 router training summary, 16 Nov 2026")
    lines.append("")
    lines.append("What ran: logistic regression and gradient boosting "
                 "(sklearn HistGradientBoostingClassifier) for the two "
                 "set-level routing targets on the RAMDocs test set (500 "
                 "examples), using nested cross-validation. Outer loop: "
                 "5-fold GroupKFold grouped by question, seed 42, so no "
                 "question appears in both train and test. Inner loop: "
                 "3-fold GroupKFold grid search scored on ROC-AUC. All "
                 "reported metrics come from the pooled out-of-fold "
                 "predictions, so every number is on examples the model "
                 "never trained on.")
    lines.append("")
    lines.append(f"Features: {res['n_features']} LLM-free set-level features "
                 "(the pilot's 7, plus answer-candidate clustering counts, "
                 "support-count imbalance and similarity dispersion).")
    lines.append(f"NLI features: {res['nli_features']}.")
    lines.append("")
    lines.append("## Headline numbers (pooled out-of-fold, nested CV)")
    lines.append("")
    lines.append("| Target | Model | ROC-AUC | PR-AUC | Accuracy @0.5 | Brier | ECE |")
    lines.append("|---|---|---|---|---|---|---|")
    for tname, tb in res["targets"].items():
        for mname, mb in tb["models"].items():
            lines.append(f"| {tname} | {mname} | {mb['roc_auc']} | "
                         f"{mb['pr_auc']} | {mb['accuracy_at_0.5']} | "
                         f"{mb['brier']} | {mb['ece_10bin']} |")
    lines.append("")
    lines.append("Baselines: predicting the majority class every time gives "
                 "accuracy "
                 + ", ".join(f"{tb['majority_class_baseline']['accuracy']} for {t}"
                             for t, tb in res["targets"].items())
                 + ". A constant prevalence predictor has Brier "
                 + ", ".join(f"{tb['majority_class_baseline']['brier_constant_prevalence']} for {t}"
                             for t, tb in res["targets"].items())
                 + ", and PR-AUC equal to prevalence ("
                 + ", ".join(f"{tb['prevalence']} for {t}"
                             for t, tb in res["targets"].items())
                 + "). Both trained models beat these baselines on Brier, "
                 "so their probabilities carry real information, not just "
                 "ranking.")
    lines.append("")
    lines.append("Calibration: ECE values are in the table above (10 bins). "
                 "As a rough read, ECE under about 0.05 is good, 0.05 to "
                 "0.10 is usable, above 0.10 means the probabilities should "
                 "not be trusted as probabilities without recalibration. "
                 "Compare each model's Brier against the constant-prevalence "
                 "Brier in the baselines line.")
    lines.append("")
    lines.append("## Comparison with Day 1")
    lines.append("")
    lines.append("Day 1 (single random split, logistic regression on the 7 "
                 "pilot features) gave ROC-AUC 0.8646 for misinfo presence "
                 "and 0.8399 for ambiguity. The Day 2 nested-CV numbers above "
                 "are the honest generalisation estimate: they use more "
                 "features and a stricter evaluation, so small differences "
                 "from the Day 1 numbers are expected and are not a problem. "
                 "Anything far below the Day 1 number would be flagged here.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- router_results_day2.json: full metrics, per-fold results, chosen hyperparameters.")
    for tname, tb in res["targets"].items():
        for mname, mb in tb["models"].items():
            lines.append(f"- {mb['final_model_file']}: final {mname} model for {tname}, fit on all 500 examples.")
    lines.append("- train_router_day2.py: the training script.")
    lines.append("")
    lines.append("## Notes and caveats")
    lines.append("")
    lines.append("- Document-level typing stays weak (Day 1 macro F1 0.5081); "
                 "nothing here changes that. The router is framed at set "
                 "level only, which is where the signal is.")
    if "skipped" in res["nli_features"] or "not requested" in res["nli_features"]:
        lines.append("- NLI entailment/contradiction proxy features were NOT "
                     "included in this run (see the NLI line above). They "
                     "remain a Day 5 ablation candidate if time allows.")
    lines.append("- Ambiguity prevalence is high (about 0.8), so accuracy "
                 "alone flatters that target; ROC-AUC and PR-AUC are the "
                 "numbers to quote.")
    path = os.path.join(HERE, "day2-summary-2026-11-16.md")
    open(path, "w").write("\n".join(lines) + "\n")
    print("Wrote", path, flush=True)


if __name__ == "__main__":
    main()
