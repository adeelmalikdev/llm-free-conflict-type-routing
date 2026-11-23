# Paper 2, bootstrap CIs and McNemar tests, 19 Nov 2026

Computed from the Day 3 per-example file, recomputed arms from
RAMDocs_test.jsonl with the same stub code as wire_generator_day3.py.
Recomputed means match day3_results.json exactly for all four arms,
so this is the same experiment, with uncertainty attached.
Method: percentile bootstrap, 10,000 resamples, seed 42, n = 500.
McNemar is the exact binomial two-sided test on exact-match
discordant pairs. Script: paper2-stats-2026-11-19.py.
Full numbers: paper2-stats-2026-11-19.json.

## Per arm, answer-set exact match and F1, 95% CI

| Arm | EM | EM 95% CI | F1 | F1 95% CI |
|---|---|---|---|---|
| Learned router | 0.454 | 0.410 to 0.496 | 0.767 | 0.745 to 0.788 |
| Oracle router | 0.524 | 0.480 to 0.568 | 0.796 | 0.774 to 0.816 |
| Always single | 0.190 | 0.156 to 0.226 | 0.647 | 0.629 to 0.665 |
| Always multi | 0.428 | 0.386 to 0.472 | 0.847 | 0.832 to 0.861 |

## Paired differences, learned router minus the other arm

| Comparison | Mean diff | 95% CI |
|---|---|---|
| EM, learned minus always multi | +0.026 | -0.008 to +0.060 |
| EM, learned minus always single | +0.264 | +0.222 to +0.306 |
| EM, learned minus oracle | -0.070 | -0.094 to -0.048 |
| F1, learned minus always multi | -0.080 | -0.099 to -0.061 |
| F1, learned minus always single | +0.120 | +0.102 to +0.138 |
| F1, learned minus oracle | -0.029 | -0.039 to -0.020 |

## McNemar on exact match

| Comparison | Learned only right | Other only right | Discordant | p (exact, two sided) |
|---|---|---|---|---|
| Learned vs always multi | 43 | 30 | 73 | 0.160 |
| Learned vs always single | 141 | 9 | 150 | < 0.001 |
| Learned vs oracle | 0 | 35 | 35 | < 0.001 |

## What this changes in the writing

The headline needs qualifying. The learned router beats always
single on exact match by a wide, significant margin (+26.4 points,
p < 0.001). Against always multi, the lead is only +2.6 points,
its 95% CI runs from -0.8 to +6.0 points, and McNemar gives
p = 0.16, so on these 500 examples with the stub generator the
router does not significantly beat always multi on exact match.
Do not write "the router beats both fixed policies" without that
qualification. The honest sentence is: routing beats always single
clearly, sits 2.6 points above always multi on exact match without
reaching significance, and sits 7.0 points below the oracle, a gap
that is significant (p < 0.001, 35 discordant pairs all in the
oracle's favour, which is expected since routing errors can only
lose against perfect routing).

The F1 split stands as written: always multi wins F1 by 8.0 points,
CI -9.9 to -6.1, so that gap is real, not noise. Routing trades
partial credit for exactness, and on this stub run the exactness
gain over always multi is small and uncertain.

What could move the always-multi comparison: a real generator
(the 8B arm) changes what FILTER and MULTI actually return, so the
2.6 point gap is a stub-generator number, not a final one. A larger
test set would also narrow the CI. Neither is available before
tomorrow's deadline unless the 8B run happens, so the paper should
report the CI and p value as they are and let the reader see the
uncertainty, rather than claiming a win over always multi.

## Limits, unchanged

Stub generator only, so FILTER and SINGLE return the same answer.
No abstention ground truth in this split. Router probabilities are
cross-fitted (OOF), so these are honest generalisation numbers.
