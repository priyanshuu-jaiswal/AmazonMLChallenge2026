# Team Task Board

Do not declare a final best model before validation.

## Priority 1 — Submission #2 diagnosis

Reproduce the exact Submission #1 algorithm on held-out training data. Measure validation F0.5, precision, recall, false positives, and false negatives. Diagnose multilingual failures, missing-address failures, and ambiguous exact-name/address failures. Record split, seed, rules, parameters, runtime, and limitations in a new experiment directory. Preserve Submission #1 artifacts.

## Priority 2 — Feasible higher-recall retrieval

Develop a rare-key candidate generator that fits available compute and storage. Do not score 12.6 billion pairs or materialize hundreds of GB. Keep frozen Stage 4 definitions unchanged; any alternative is a separately documented experiment.

## Priority 3 — Better pair scorer

Evaluate practical pair features and scorers after retrieval is computationally feasible. Measure macro F0.5, including singleton behavior.

## Priority 4 — Validate Submission #2

Check full test S1 coverage, ID validity, duplicate/self-match rules, candidate containment when supplied, and run the official validator.

## Priority 5 — Submit to leaderboard

Submit only after evaluation and output validation. Record the submission and score without overwriting prior results.
