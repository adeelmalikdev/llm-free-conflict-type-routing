"""
Pilot: Can cheap, LLM-free features detect conflict structure in RAMDocs?

Riskiest assumption of the proposed project: the conflict structure of a
retrieved set (document types; set-level ambiguity / misinformation) is
separable with small, CPU-only models BEFORE any LLM generation.

Data: RAMDocs test split (Wang et al. 2025, arXiv:2504.13079), MIT licence,
https://github.com/HanNight/RAMDocs -- 500 examples, documents labelled
correct / misinfo / noise, with per-document answers, disambiguated entities,
gold and wrong answers.

Experiments (group split by question, seed 42, no leakage across splits):
  P1  Document-type classification (correct/misinfo/noise) from TF-IDF of
      question+document text, logistic regression.
  P2  Set-level routing signals from handcrafted, LLM-free features only
      (answer diversity, TF-IDF doc-doc similarity stats, doc-length stats):
      (a) does the set contain misinformation?  (b) is the query ambiguous
      (multi-entity)?  Logistic regression, report accuracy / ROC-AUC.
  P3  Cheap answer-selection reference: majority-answer among documents
      (exact-match vs gold set) -- the floor any router+generator must beat.

Run: python pilot_ramdocs_router.py /path/to/RAMDocs_test.jsonl
Outputs: pilot_results.json in the current directory.
"""
import json, sys, collections, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

path = sys.argv[1] if len(sys.argv) > 1 else "RAMDocs_test.jsonl"
rows = [json.loads(l) for l in open(path)]
rng = np.random.RandomState(42)
results = {"dataset": "RAMDocs test (500 examples)", "seed": 42}

# ---------- P1: document-type classification ----------
docs = []
for i, r in enumerate(rows):
    for d in r["documents"]:
        docs.append((i, r["question"] + " " + d["text"][:2000], d["type"]))
groups = np.array([d[0] for d in docs])
X_text = [d[1] for d in docs]
y = np.array([d[2] for d in docs])
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
tr, te = next(gss.split(X_text, y, groups))
vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
Xtr = vec.fit_transform([X_text[i] for i in tr])
Xte = vec.transform([X_text[i] for i in te])
clf = LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced")
clf.fit(Xtr, y[tr])
pred = clf.predict(Xte)
results["P1_doc_type"] = {
    "n_train_docs": int(len(tr)), "n_test_docs": int(len(te)),
    "accuracy": round(float(accuracy_score(y[te], pred)), 4),
    "macro_f1": round(float(f1_score(y[te], pred, average="macro")), 4),
    "per_class": classification_report(y[te], pred, output_dict=True, zero_division=0),
    "majority_baseline_accuracy": round(float(max(collections.Counter(y[te]).values()) / len(te)), 4),
}

# ---------- P2: set-level routing features (LLM-free) ----------
def norm_ans(a):
    return re.sub(r"\s+", " ", a.strip().lower())

def set_features(r):
    texts = [d["text"] for d in r["documents"]]
    answers = [norm_ans(d["answer"]) for d in r["documents"] if norm_ans(d["answer"]) not in ("unknown", "")]
    v = TfidfVectorizer(max_features=5000)
    try:
        M = v.fit_transform(texts)
        sims = (M @ M.T).toarray()
        iu = np.triu_indices(len(texts), k=1)
        pair = sims[iu] if len(texts) > 1 else np.array([0.0])
    except ValueError:
        pair = np.array([0.0])
    cnt = collections.Counter(answers)
    top_share = (max(cnt.values()) / len(answers)) if answers else 0.0
    return [
        len(texts),                                   # n docs
        float(np.mean([len(t.split()) for t in texts])),
        float(pair.mean()), float(pair.min()),        # doc-doc similarity
        len(cnt),                                     # distinct answers
        top_share,                                    # majority answer share
        float(len(answers)),                          # docs with an answer
    ]

F = np.array([set_features(r) for r in rows], dtype=float)
amb = np.array([1 if len(r["disambig_entity"]) > 1 else 0 for r in rows])
mis = np.array([1 if any(d["type"] == "misinfo" for d in r["documents"]) else 0 for r in rows])
idx = np.arange(len(rows)); rng.shuffle(idx)
cut = int(0.75 * len(rows)); tr_i, te_i = idx[:cut], idx[cut:]
out = {}
for name, target in [("misinfo_present", mis), ("ambiguous_multi_entity", amb)]:
    c = LogisticRegression(max_iter=1000, class_weight="balanced")
    c.fit(F[tr_i], target[tr_i])
    p = c.predict_proba(F[te_i])[:, 1]
    pr = (p > 0.5).astype(int)
    out[name] = {
        "prevalence_test": round(float(target[te_i].mean()), 4),
        "accuracy": round(float(accuracy_score(target[te_i], pr)), 4),
        "roc_auc": round(float(roc_auc_score(target[te_i], p)), 4),
        "majority_baseline_accuracy": round(float(max(target[te_i].mean(), 1 - target[te_i].mean())), 4),
    }
results["P2_set_level"] = out

# ---------- P3: majority-answer reference ----------
def norm_set(xs):
    return set(norm_ans(x) for x in xs)
hits_all, hits_any, n = 0, 0, 0
for r in rows:
    answers = [norm_ans(d["answer"]) for d in r["documents"] if norm_ans(d["answer"]) not in ("unknown", "")]
    if not answers:
        continue
    top = collections.Counter(answers).most_common(1)[0][0]
    gold = norm_set(r["gold_answers"])
    n += 1
    hits_any += 1 if top in gold else 0
    hits_all += 1 if gold == {top} else 0
results["P3_majority_answer"] = {
    "n": n,
    "top_answer_in_gold_rate": round(hits_any / n, 4),
    "top_answer_equals_full_gold_set_rate": round(hits_all / n, 4),
    "note": "Single-answer majority vote cannot return multi-answer gold sets; gap quantifies why an ambiguity route is needed.",
}
print(json.dumps(results, indent=2))
json.dump(results, open("pilot_results.json", "w"), indent=2)
