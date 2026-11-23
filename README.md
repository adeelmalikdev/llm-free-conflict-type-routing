# LLM-Free Conflict-Type Routing for Conflicting-Evidence RAG

Code, results, figures and paper for a study of a simple question: when a retrieval system pulls back a set of documents that disagree with each other, can you predict the *type* of conflict from the set itself, cheaply, with no language model in the loop, and use that prediction to decide how to answer?

Status: submitted to the AAAI-27 WMAC workshop (Workshop on Measuring Collective Intelligence in Agentic Systems), under review. Notification is due 2 December 2026. This repository is private while the paper is under review, and the paper PDF is the anonymous submission version.

## The idea

Retrieval-augmented generation systems usually pick one answering strategy and use it for everything: answer from a single pass, or pay for a multi-agent debate style procedure on every question. Both waste something. Single-pass answering fails on conflicting evidence. Multi-agent procedures cost time and tokens on questions that never needed them.

This work treats the choice as a routing problem:

1. From the retrieved document set alone, compute 22 set-level features. No LLM calls, no embeddings from a hosted model, just counts, agreement structure and answer-candidate statistics over the set.
2. Train a classifier to predict the conflict type in the set: misinformation present, or ambiguous with several defensible answers.
3. Route each question to a strategy: SINGLE (answer once), MULTI (produce and reconcile several answers), FILTER (answer from a cleaned subset) or ABSTAIN.

The router is a measurement instrument as much as a system: it measures when a collective answering procedure actually earns its cost.

## Verified results

### Day 2: the router itself

Nested cross-validation, GroupKFold grouped by question so no question leaks across folds, 22 LLM-free set-level features, all 500 RAMDocs examples.

- Misinformation detection, gradient boosting: ROC-AUC **0.9531**, PR-AUC **0.9552**.
- Ambiguity detection, gradient boosting: ROC-AUC **0.9179**, PR-AUC **0.9754**.

### Day 3: end-to-end routing

All 500 RAMDocs examples, cross-fitted router probabilities, and a deterministic extractive stub generator standing in for the answering model, so the routing effect is isolated from generator quality.

- Learned router: exact match **0.454**, F1 **0.7669**.
- Always-single: exact match **0.190**.
- Always-multi: exact match **0.428**, F1 **0.8467**.
- Oracle router: exact match **0.524**.

### Paired statistics

Percentile bootstrap, 10,000 resamples, plus exact McNemar tests. Script and full output are in `results/`.

- Learned vs always-single: **+26.4 exact-match points**, 95% CI +22.2 to +30.6, p < 0.001. This is the solid win.
- Learned vs always-multi: +2.6 points, CI -0.8 to +6.0, p = 0.16. That is a tie, and the paper reports it as a tie. Always-multi also wins on F1 by 8.0 points.

## Honest limits

- With the stub generator, the FILTER and SINGLE routes return the same answer, because filtering only changes what a real generator sees. So no filter gain is claimed anywhere in this work.
- The RAMDocs split used here has no unanswerable examples, so the ABSTAIN route has no ground truth yet. Testing abstention needs an unanswerable split, for example from FaithEval.
- The gap to the oracle router is about 7 exact-match points, so the router is good, not perfect.

## Repository layout

- `src/`
  - `pilot_ramdocs_router.py` - the Day 1 pilot: document typing and set-level analysis of RAMDocs.
  - `train_router_day2.py` - feature extraction and the nested-CV router training.
  - `wire_generator_day3.py` - the end-to-end routing run with the stub generator.
  - `paper2-stats-2026-11-19.py` - bootstrap confidence intervals and McNemar tests.
  - `make_figs_v2.py` - the final paper figures.
- `results/` - `router_results_day2.json`, `day3_results.json`, the stats JSON and write-up, and the Day 2 and Day 3 summaries.
- `figures/` - the two figures in the paper: per-strategy EM and F1, and the paired differences.
- `paper/` - the submitted paper PDF (anonymous, 4 pages) and the full draft text it was built from.
- `optional/` - the 8B generator arm: `run_8b_cached.py` and `paper2-8b-colab.ipynb`. **This arm was not part of the workshop submission and has not been run to completion.** It exists so the stub generator can later be swapped for a real local model (Llama 3.1 8B Instruct, Q4_K_M GGUF, through llama-cpp). A smoke test on a small shared CPU VM measured about 95 seconds for one short call, which makes the full run impractical there; the Colab notebook is the realistic route. Extra dependency for this folder only: `optional/requirements-8b.txt`.

## Data

The dataset is RAMDocs, from the paper "RAMDocs: A Benchmark for Retrieval-Augmented Generation with Conflicting Evidence" (arXiv:2504.13079). The RAMDocs test file is not redistributed here; download it from the original source and place it as `RAMDocs_test.jsonl` next to the scripts if you rerun the pipeline. References for the related benchmarks discussed in the paper: CRAG (arXiv:2401.15884), FaithEval (arXiv:2410.03727) and AmbigDocs (arXiv:2404.12447).

## Running it

```bash
pip install -r requirements.txt
# place RAMDocs_test.jsonl in src/ first (see Data above)
cd src
python train_router_day2.py
python wire_generator_day3.py
python paper2-stats-2026-11-19.py
```

Requirements for the main pipeline: numpy, scikit-learn, joblib, matplotlib. Everything runs on CPU.

## Citation

Muhammad Adeel Tariq. "LLM-Free Conflict-Type Routing for Conflicting-Evidence RAG." Submitted to the AAAI-27 WMAC workshop, November 2026.

## License

MIT, see `LICENSE`. RAMDocs and the other benchmarks belong to their original authors.
