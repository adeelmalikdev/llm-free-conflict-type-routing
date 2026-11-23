# Paper 2, Day 3 generator wiring summary, 18 Nov 2026

Router: gradient_boosting (saved Day 2 models). Generator: stub. Router probs: oof. n = 500. Thresholds: misinfo 0.5, ambiguous 0.5.

Route accuracy vs true labels: 0.838.

## Arms (answer-set EM / F1)

- learned-router: EM 0.454, F1 0.7669, routes {'SINGLE': 52, 'MULTI': 205, 'FILTER': 242, 'ABSTAIN': 1}, median latency 1.9 ms, mean calls 0.998, mean tokens 0.0
- oracle-router: EM 0.524, F1 0.7955, routes {'SINGLE': 43, 'FILTER': 243, 'ABSTAIN': 1, 'MULTI': 213}, median latency 0.0 ms, mean calls 0.998, mean tokens 0.0
- always-single: EM 0.19, F1 0.6473, routes {'SINGLE': 500}, median latency 0.0 ms, mean calls 1.0, mean tokens 0.0
- always-multi: EM 0.428, F1 0.8467, routes {'MULTI': 500}, median latency 0.0 ms, mean calls 1.0, mean tokens 0.0

## Notes and caveats

- Router probabilities are cross-fitted (OOF) by default, so the learned-router numbers above are honest. The in-sample mode (saved Day 2 final models on the same 500) gives route accuracy 0.998 and must not be quoted.
- With the stub generator, FILTER and SINGLE return the same answer, because filtering only changes what a real generator is shown. The FILTER gain can only be measured with the llama-cpp arm.
- Read EM and F1 together: the learned router beats always-multi on exact set match, but always-multi scores higher on F1, because returning every cluster usually includes the gold answers even when it adds wrong ones. That is the cost of routing ambiguous sets to a single answer, and the 8B arm is where it gets tested properly.
- gold_answers is never empty in this 500-example split, so abstention has no ground-truth class here; ABSTAIN fires only when no document carries an answer. Real abstention measurement needs the FaithEval-unanswerable split.
- The llama-cpp arm is written but untested on this VM: no GGUF model and no llama-cpp-python are installed here (checked 18 Nov 2026). Running it needs a model download first, plus a timing run before the full 500.
- Day 1 majority-answer floor for reference: a single-answer majority vote matches the full gold set only 19% of the time, which is why the MULTI route matters.
