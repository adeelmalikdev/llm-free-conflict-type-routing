# Paper 2 draft, 20 Nov 2026: LLM-free conflict-type routing on RAMDocs

Status: full draft text for a 4-page WMAC short paper, written tonight from the corrected submission pack. Nothing here is copied from the superseded 18 Nov claims draft. Every number below comes from paper2-stats-2026-11-19.json (bootstrap, 10,000 resamples, seed 42) or the Day 2 and Day 3 summaries. The one sentence the paper must never write is in the limits section at the end.

Deadline: WMAC 2027, AAAI-27 Workshop on Multi-Agent Collaboration, 23:59 AoE Fri 20 Nov 2026, which is 16:59 PKT Sat 21 Nov 2026. Submit on OpenReview at https://openreview.net/group?id=AAAI.org/2027/Workshop/WMAC . Notification 2 Dec 2026. Non-archival.

What remains after this draft: paste into the AAAI-27 author kit, embed the two figures, check the page count stays at 4 pages excluding references, run the double-blind pass (no author names, affiliations, self-identifying text, personal repo links, or identifying PDF metadata), then submit and save the OpenReview submission number.

References verified 21 Nov 2026 by web search against the arXiv pages: CRAG is Yan, Gu, Zhu and Ling, arXiv:2401.15884; FaithEval is Ming et al., arXiv:2410.03727, ICLR 2025; AmbigDocs is Lee, Ye and Choi, arXiv:2404.12447, COLM 2024. The RAMDocs citation was verified 20 Nov (Wang, Prasad, Stengel-Eskin and Bansal, arXiv:2504.13079, April 2025). All four are now full citations in the References section below.

---

Title: Do You Need the Debate? LLM-Free Conflict-Type Routing Predicts When Multi-Agent RAG Helps, Filters, or Should Abstain

Abstract: Retrieval with conflicting evidence forces a choice about the answer contract: give one answer, give every answer, filter first, or say nothing. We test whether that choice can be made without calling a language model at all. A CPU-only router built from 22 set-level features predicts, on the 500-example RAMDocs test set, whether a retrieved set contains misinformation (ROC-AUC 0.953) and whether it carries more than one valid answer (ROC-AUC 0.918). Wired to a deterministic stub generator, the learned router reaches exact set match 0.454, against 0.190 for always answering once and 0.428 for always answering every cluster. The gain over always-single is solid (plus 26.4 points, 95% CI plus 22.2 to plus 30.6, p under 0.001). The lead over always-multi is not (plus 2.6 points, CI minus 0.8 to plus 6.0, p = 0.16), and always-multi still wins on F1 by 8.0 points. The router sits 7.0 points below an oracle router, so better routing still has headroom. The router itself adds a median 1.9 ms of CPU per example and no language model calls. Filter and abstention gains remain unmeasured: with the stub generator, filtering changes nothing a reader can see, and this split has no unanswerable questions.

## 1. Introduction

Retrieval-augmented generation usually assumes the retrieved documents agree, or that one of them is right. In practice they often do not. A query can be ambiguous, so two documents support two different answers, both correct for different readings. A query can also draw in misinformation, so one document supports an answer that is simply wrong. Noise makes both harder to see. When the retrieved set contains these conflicts, the right move depends on which conflict is present. An ambiguous set calls for every valid answer. A set with misinformation calls for filtering before answering. A set with nothing to support an answer calls for abstaining. Giving one answer in all three cases is wrong in two of them.

The recent answer to this problem has been debate. In MADAM-RAG, language model agents argue over the merits of each answer across several rounds, and an aggregator collects the surviving answers while discarding misinformation and noise (Wang et al., 2025). It works, and it is expensive: several model calls per query, several rounds, and an aggregator on top. That cost raises a narrower question, and it is the question this paper asks. How much of the routing decision can be made before any model is called, from cheap signals in the retrieved set itself?

We build a router that never calls a language model. It reads 22 features computed from the retrieved documents and their answers: how many distinct answers there are, how support is split across them, how similar the documents are to each other, and how the answers cluster. Two gradient boosting models turn those features into two probabilities: the chance the set contains misinformation, and the chance it carries more than one valid answer. A fixed rule maps the two probabilities to one of four routes: SINGLE, MULTI, FILTER, or ABSTAIN. The router is evaluated on RAMDocs, a 500-query dataset built to combine ambiguity, misinformation and noise in a single query (Wang et al., 2025). The strongest reported baseline on it, Llama-3.3-70B-Instruct, reaches only 32.6 exact match, which shows how hard the combined conflicts are.

Our contribution is a measurement, not a system claim. We report how well conflict type can be predicted from cheap features, what that prediction buys end to end against fixed answering policies, and where the measurement stops. The headline is mixed, and we report it as mixed. Routing beats always answering once by a wide and significant margin. Against always answering every cluster, the exact match lead is small and not significant, while the F1 lead runs the other way. The filter and abstention routes are not measured here at all, because the generator in this run is a stub and the split has no unanswerable questions. A short paper is the right shape for that result: an early, honest measurement with its limits stated, not a full system.

## 2. Related work

Three lines of work frame this paper. The first is retrieval that corrects itself. CRAG (Yan et al., 2024) adds a small evaluator that grades retrieved documents and triggers more retrieval when the grades are low. Our router differs in what it routes on and what routing changes. CRAG routes on relevance quality to retrieve more. We route on conflict semantics, misinformation against ambiguity, to change the answer contract itself: one answer, several answers, filter first, or abstain. Our router is also language model free, where CRAG's evaluator is a fine-tuned model. A relevance router baseline in the CRAG style is the natural next comparison, and we name it as future work rather than claiming it here.

The second line is the dataset and the debate system built on it. RAMDocs simulates realistic retrieval conditions with multiple valid answers, varied document support, inter-document conflict, misinformation and noise, over 500 unique queries drawn from AmbigDocs (Wang et al., 2025). MADAM-RAG debates over the answers across rounds and aggregates the survivors. Our work uses their dataset and their framing of the conflict types, but replaces the debate with a feature router, to ask how much of the decision the debate was doing can be done for free. We do not run MADAM-RAG ourselves, so we claim no cost saving against debate. That comparison needs a debate baseline with measured cost, and it does not exist in this run.

The third line is evaluation that isolates one conflict at a time. FaithEval (Ming et al., 2024) measures misinformation handling, including unanswerable cases, and AmbigDocs (Lee et al., 2024) measures ambiguity. RAMDocs combines them, which is why we use it. But the combination also means this split inherits a gap: like the RAMDocs test set we use, neither single-conflict dataset on its own gives us an abstention measurement here. Our abstention route is defined and implemented, but it fires once in 500 examples, and there is no ground truth class for it in this split. Measuring it properly needs the FaithEval unanswerable split, which we leave for the next run.

## 3. Method

### 3.1 Data

We use the RAMDocs test set: 500 examples, each with a question, a set of retrieved documents, a disambiguating entity, gold answers, and wrong answers. Documents carry labels of correct, misinformation, or noise. The set is grouped by question, seed 42, so no question appears in both training and test folds. Two prevalence numbers shape everything that follows. Misinformation is present in 48.6% of examples, so a majority-class guess gets 51.4% accuracy on that target. Ambiguity, meaning more than one valid answer, is present in about 80% of examples, so accuracy alone flatters any model on that target, and ROC-AUC and PR-AUC are the numbers to trust. A single-answer majority vote matches the full gold set only 19% of the time, which is why returning every cluster matters at all. Document-level typing, classifying each document on its own, stays weak: macro F1 0.508 in our Day 1 run. The router is therefore framed at set level only, where the signal is, and we do not claim document-level results.

### 3.2 Features

The router uses 22 language model free features computed from each retrieved set. They fall into four groups. Document similarity: mean, minimum, maximum and standard deviation of pairwise TF-IDF cosine similarity between documents. Answer support: the number of distinct answers, the share held by the majority answer, the top answer count, the second answer count, their difference and ratio, and the count of answers with support of at least two. Answer clustering: distinct answers are grouped greedily by token overlap (Jaccard 0.5 or above joins an existing cluster), giving the number of clusters, the largest cluster support and share, the number of singleton clusters, and the mean cluster size. Answer shape: answer entropy over the support distribution, the number of documents carrying an answer, the fraction carrying none, the mean document length in words, and the mean answer length in tokens. A planned fifth group, pairwise entailment and contradiction proxies from a small NLI model, was not included: the libraries it needs were not available on the machine used for this run. That is a stated gap, not a silent one, and it is a candidate ablation for later work.

### 3.3 Router

Two gradient boosting models (scikit-learn HistGradientBoostingClassifier) predict the two targets: misinformation present, and ambiguous with multiple entities. Logistic regression is trained the same way as a comparison. Evaluation is nested cross-validation with grouped folds: an outer 5-fold GroupKFold grouped by question, and an inner 3-fold GroupKFold grid search scored on ROC-AUC, seed 42 throughout. All reported router metrics come from pooled out-of-fold predictions, so every number is measured on examples the model never trained on. This matters, because the final models refit on all 500 examples reach route accuracy 0.998 on those same examples. That is memorisation. Only the cross-fitted numbers in this paper are quotable, and the in-sample numbers appear nowhere else in it.

Calibration is reported alongside ranking because the router's probabilities are used as thresholds, not only as a ranking. Expected calibration error uses 10 bins. As a rough read, ECE under about 0.05 is good, 0.05 to 0.10 is usable, and above 0.10 the probabilities should not be trusted as probabilities without recalibration.

### 3.4 Routing rule

Thresholds are 0.5 on both targets. The rule, in priority order, is: ABSTAIN if no document carries an answer at all; else FILTER if the misinformation probability reaches 0.5; else MULTI if the ambiguity probability reaches 0.5; else SINGLE. Misinformation safety comes first, then ambiguity. The oracle router uses the same rule with true labels instead of predicted probabilities, so the learned-versus-oracle comparison is apples to apples: the only difference is routing error.

The generator in this run is a deterministic stub, no language model at all. SINGLE and FILTER return the majority answer. MULTI returns one representative answer per token cluster. ABSTAIN returns no answer. With the stub, FILTER and SINGLE return the same answer, because filtering only changes what a real generator is shown. The filter gain can therefore only be measured with a real generator, and this paper claims none. A local quantised generator arm (an 8B model through llama-cpp) is written but untested on the machine used here: no model file and no llama-cpp install were available, so running it needs a download plus a timing run before the full 500. That arm is the single most valuable next step, and it is the reason this is a short paper rather than a full one.

### 3.5 Metrics and statistics

End-to-end quality is answer-set exact match (the returned set equals the gold set exactly) and answer-set F1 (partial credit). Exact match is the primary measure, because it is where the routing decision shows: returning one answer when several are valid, or several when one is, fails exact match even when F1 is generous. Uncertainty is attached by percentile bootstrap, 10,000 resamples, seed 42, on all 500 examples, and by McNemar exact binomial two-sided tests on exact-match discordant pairs. Cost is reported only as measured: router latency and the number of language model calls the router itself makes, which is zero by construction.

## 4. Results

### 4.1 The router predicts conflict type well

Table 1 gives the router numbers. Gradient boosting reaches ROC-AUC 0.953 on misinformation presence and 0.918 on ambiguity, with PR-AUC 0.955 and 0.975. Calibration is good on both targets (ECE 0.039 and 0.046), so the 0.5 thresholds are defensible as probabilities, not only as cut points. Logistic regression is clearly weaker on both targets, which is why the end-to-end run uses gradient boosting. Both trained models beat their constant-prevalence baselines on Brier score, so the probabilities carry real information rather than ranking alone.

Table 1. Router performance, nested cross-validation, pooled out-of-fold predictions, n = 500.

| Target | Model | ROC-AUC | PR-AUC | Accuracy at 0.5 | Brier | ECE |
|---|---|---|---|---|---|---|
| Misinformation present | Gradient boosting | 0.953 | 0.955 | 0.876 | 0.084 | 0.039 |
| Misinformation present | Logistic regression | 0.869 | 0.885 | 0.782 | 0.148 | 0.084 |
| Ambiguous, multi entity | Gradient boosting | 0.918 | 0.975 | 0.900 | 0.080 | 0.046 |
| Ambiguous, multi entity | Logistic regression | 0.882 | 0.967 | 0.836 | 0.112 | 0.063 |

Prevalence: misinformation 0.486, ambiguity about 0.80. Majority-class accuracy is 0.514 and 0.80. Constant-prevalence Brier is 0.250 and 0.160.

### 4.2 End to end, routing beats always-single clearly and ties always-multi

Table 2 and Figure 1 give the end-to-end numbers with the stub generator and cross-fitted router probabilities. The learned router reaches exact match 0.454 (95% CI 0.410 to 0.496) and F1 0.767 (0.745 to 0.788). Against always-single, exact match 0.190 (0.156 to 0.226), the gain is plus 26.4 points, CI plus 22.2 to plus 30.6, McNemar p under 0.001, with 141 examples the router gets right and always-single gets wrong against 9 the other way. That win is solid, and it is the paper's one unqualified claim.

Against always-multi, exact match 0.428 (0.386 to 0.472), the lead is plus 2.6 points, CI minus 0.8 to plus 6.0, McNemar p = 0.16, with 43 discordant pairs in the router's favour against 30 the other way. On these 500 examples with the stub generator, that is a tie, not a win. Figure 2 shows it directly: the always-multi bar crosses zero. The oracle router reaches 0.524 (0.480 to 0.568), so the learned router sits 7.0 points below perfect routing, CI minus 9.4 to minus 4.8, p under 0.001, all 35 discordant pairs in the oracle's favour. Better routing still has headroom, and the gap is the size of the prize for the next iteration.

Table 2. End-to-end routing, cross-fitted router probabilities, n = 500, stub generator.

| Arm | Exact match | EM 95% CI | F1 | F1 95% CI | Routes (SINGLE, MULTI, FILTER, ABSTAIN) |
|---|---|---|---|---|---|
| Learned router | 0.454 | 0.410 to 0.496 | 0.767 | 0.745 to 0.788 | 52, 205, 242, 1 |
| Oracle router | 0.524 | 0.480 to 0.568 | 0.796 | 0.774 to 0.816 | 43, 213, 243, 1 |
| Always-single | 0.190 | 0.156 to 0.226 | 0.647 | 0.629 to 0.665 | 500, 0, 0, 0 |
| Always-multi | 0.428 | 0.386 to 0.472 | 0.847 | 0.832 to 0.861 | 0, 500, 0, 0 |

Route accuracy of the learned router against true labels is 0.838.

Figure 1. Exact match and F1 per arm, with 95% bootstrap confidence intervals. File: paper2-fig1-em-f1-2026-11-20.png. Caption must state n = 500, stub generator, cross-fitted router probabilities, and percentile bootstrap with 10,000 resamples, seed 42.

Figure 2. Paired exact match differences, learned router minus each other arm, with 95% confidence intervals. File: paper2-fig2-paired-diff-2026-11-20.png. The always-multi bar crosses zero: the reader can see the tie.

### 4.3 The F1 split, and what it means

The F1 numbers run the other way. Always-multi wins F1 by 8.0 points over the learned router (CI minus 9.9 to minus 6.1), and the learned router beats always-single on F1 by 12.0 points (CI plus 10.2 to plus 13.8). This is not a contradiction. It is the two measures rewarding different things. Returning every answer cluster usually includes the gold answers along with wrong ones, so F1 rewards over-answering. Exact match rewards getting the set exactly right, which is what routing is for. The honest reading is that routing trades partial credit for exactness, and on this stub run the exactness gain over always-multi is small and uncertain. A real generator could move both numbers: it changes what FILTER and MULTI actually return, so the 2.6 point gap is a stub-generator number, not a final one. A larger test set would also narrow the confidence interval. Neither is available before this deadline, so the paper reports the interval and the p value as they are and lets the reader see the uncertainty.

### 4.4 Cost, stated narrowly

What is measured: the router is CPU only, no language model calls by construction, median latency 1.9 ms per example in the Day 3 run, mean 0.998 calls per example (the stub counts as a call in the logging, and it logs zero tokens because it is a stub). What is not measured: no debate baseline was run, so this paper claims no cost saving against debate. The cost claim it can make is narrower and true: the router itself adds about 2 ms of CPU and no language model calls, and it avoids returning every answer cluster on the 295 examples it does not route to MULTI. The token saving of that avoidance is unmeasured with the stub and is stated as such.

## 5. Limitations and next steps

Four limits bound this paper, and each one names its fix. First, the stub generator. FILTER and SINGLE return the same answer, so the filter route is defined and routed to 242 times but its gain is unmeasured. The fix is the written 8B generator arm: download a quantised model, run a timing pass, then the full 500, and report the filter gain and whether the F1 gap closes. Second, abstention. No example in this split has an empty gold set, ABSTAIN fires once, and there is no ground truth class for it. The fix is the FaithEval unanswerable split as a transfer test. Third, the router itself. NLI entailment and contradiction proxy features were planned and not included, and no ablation of feature groups was run. The fix is a Day 5 style ablation: feature groups, and a language model judge router on a subsample as a comparison point. Fourth, scale. Five hundred examples give wide intervals: the always-multi comparison spans minus 0.8 to plus 6.0 points. A larger test set, or a second dataset, would narrow that interval and test whether the tie holds. A full paper would need the first two fixes plus the ablations and a debate baseline with measured cost. None of those exist yet, which is why this submission is a short paper.

## 6. Conclusion

A language model free router can tell, from cheap features of a retrieved set, whether it contains misinformation and whether it carries more than one valid answer, with ROC-AUC 0.953 and 0.918 on RAMDocs. Wired to a stub generator, that prediction beats always answering once by 26.4 points on exact match, ties always answering every cluster at plus 2.6 points without significance, and sits 7 points below an oracle router. The cost of the router itself is about 2 ms of CPU and no model calls. The result is useful in two ways. Practically, it says a cheap pre-check can replace the single-answer default and capture most of what perfect routing would give, while the multi-answer default remains a strong and cheap competitor on this data. Scientifically, it says conflict type is feature-detectable at set level even where document-level typing is weak, and the open question is now what a real generator does with the filter route. That is the next run, and this paper is written so that run can replace the stub numbers without rewriting the claim.

## References

Lee, Y., Ye, X., and Choi, E., 2024. AmbigDocs: Reasoning across Documents on Different Entities under the Same Name. arXiv:2404.12447. Published as a conference paper at COLM 2024.

Ming, Y., Purushwalkam, S., Pandit, S., Ke, Z., Nguyen, X.-P., Xiong, C., and Joty, S., 2024. FaithEval: Can Your Language Model Stay Faithful to Context, Even If "The Moon is Made of Marshmallows". arXiv:2410.03727. Published as a conference paper at ICLR 2025.

Wang, H., Prasad, A., Stengel-Eskin, E., and Bansal, M., 2025. Retrieval-Augmented Generation with Conflicting Evidence. arXiv:2504.13079. RAMDocs dataset and MADAM-RAG. Data and code at https://github.com/HanNight/RAMDocs .

Yan, S.-Q., Gu, J.-C., Zhu, Y., and Ling, Z.-H., 2024. Corrective Retrieval Augmented Generation. arXiv:2401.15884.

---

## Limits this draft obeys (do not edit them out)

- Do not claim the router beats always-multi on exact match. Report plus 2.6 points, CI minus 0.8 to plus 6.0, p = 0.16.
- Do not claim the router wins on F1. Always-multi wins F1 by 8.0 points.
- Do not claim a filter gain. With the stub generator FILTER and SINGLE return the same answer.
- Do not claim an abstention result. No example in this split has an empty gold set. ABSTAIN fired once.
- Do not quote the in-sample router numbers. Saved Day 2 final models on the same 500 give route accuracy 0.998. That is memorisation. Only cross-fitted numbers are quotable.
- Do not claim a debate cost saving. No debate baseline was run.
- Do not quote zero tokens as a cost result. The stub logs zero tokens because it is a stub.
- Do not copy the 18 Nov claims draft sentence that the router beats both fixed policies. It was corrected by the 19 Nov stats.
