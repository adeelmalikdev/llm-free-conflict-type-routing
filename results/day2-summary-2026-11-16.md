# Paper 2, Day 2 router training summary, 16 Nov 2026

What ran: logistic regression and gradient boosting (sklearn HistGradientBoostingClassifier) for the two set-level routing targets on the RAMDocs test set (500 examples), using nested cross-validation. Outer loop: 5-fold GroupKFold grouped by question, seed 42, so no question appears in both train and test. Inner loop: 3-fold GroupKFold grid search scored on ROC-AUC. All reported metrics come from the pooled out-of-fold predictions, so every number is on examples the model never trained on.

Features: 22 LLM-free set-level features (the pilot's 7, plus answer-candidate clustering counts, support-count imbalance and similarity dispersion).
NLI features: requested but skipped: transformers/torch not available (ModuleNotFoundError).

## Headline numbers (pooled out-of-fold, nested CV)

| Target | Model | ROC-AUC | PR-AUC | Accuracy @0.5 | Brier | ECE |
|---|---|---|---|---|---|---|
| misinfo_present | logreg | 0.8692 | 0.8851 | 0.782 | 0.1476 | 0.0844 |
| misinfo_present | gradient_boosting | 0.9531 | 0.9552 | 0.876 | 0.0844 | 0.0391 |
| ambiguous_multi_entity | logreg | 0.8819 | 0.9672 | 0.836 | 0.1119 | 0.0627 |
| ambiguous_multi_entity | gradient_boosting | 0.9179 | 0.9754 | 0.9 | 0.0802 | 0.0456 |

Baselines: predicting the majority class every time gives accuracy 0.514 for misinfo_present, 0.8 for ambiguous_multi_entity. A constant prevalence predictor has Brier 0.2498 for misinfo_present, 0.16 for ambiguous_multi_entity, and PR-AUC equal to prevalence (0.486 for misinfo_present, 0.8 for ambiguous_multi_entity). Both trained models beat these baselines on Brier, so their probabilities carry real information, not just ranking.

Calibration: ECE values are in the table above (10 bins). As a rough read, ECE under about 0.05 is good, 0.05 to 0.10 is usable, above 0.10 means the probabilities should not be trusted as probabilities without recalibration. Compare each model's Brier against the constant-prevalence Brier in the baselines line.

## Comparison with Day 1

Day 1 (single random split, logistic regression on the 7 pilot features) gave ROC-AUC 0.8646 for misinfo presence and 0.8399 for ambiguity. The Day 2 nested-CV numbers above are the honest generalisation estimate: they use more features and a stricter evaluation, so small differences from the Day 1 numbers are expected and are not a problem. Anything far below the Day 1 number would be flagged here.

## Files

- router_results_day2.json: full metrics, per-fold results, chosen hyperparameters.
- router_final_logreg_misinfo_present.joblib: final logreg model for misinfo_present, fit on all 500 examples.
- router_final_gradient_boosting_misinfo_present.joblib: final gradient_boosting model for misinfo_present, fit on all 500 examples.
- router_final_logreg_ambiguous_multi_entity.joblib: final logreg model for ambiguous_multi_entity, fit on all 500 examples.
- router_final_gradient_boosting_ambiguous_multi_entity.joblib: final gradient_boosting model for ambiguous_multi_entity, fit on all 500 examples.
- train_router_day2.py: the training script.

## Notes and caveats

- Document-level typing stays weak (Day 1 macro F1 0.5081); nothing here changes that. The router is framed at set level only, which is where the signal is.
- NLI entailment/contradiction proxy features were NOT included in this run (see the NLI line above). They remain a Day 5 ablation candidate if time allows.
- Ambiguity prevalence is high (about 0.8), so accuracy alone flatters that target; ROC-AUC and PR-AUC are the numbers to quote.
