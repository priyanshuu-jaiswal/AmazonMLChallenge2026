# Team Task Board

Do not declare a final best model before validation.

## Current project status

- Stages 1–3: complete.
- Stage 4: complete and frozen.
- Stage 5: incomplete; the initial scorer and Submission #1 exist.
- Submission #1: fast direct indexed exact/strong-match baseline; leaderboard F0.5=0.228. This is not the Stage 5 scorer's score.
- Immediate objective: evaluate Submission #1 and produce a validated Submission #2.

## Frozen Stage 4 reference

Stage 4.7 used 1,500 S1 entities and 5,156 true links:

| Strategy | Recalled links | Recall |
|---|---:|---:|
| A = Prefix5 ∪ Address2500 | 4,959/5,156 | 96.18% |
| A + HF-address/name (selected) | 5,023/5,156 | 97.42% |
| B = Prefix5 ∪ Address25000 | 5,096/5,156 | 98.84% |
| A + name-token≤2,500 | 5,034/5,156 | 97.63% |

The frozen expansion requires same-country address token df 2,501–25,000 AND same-country name token df 1–25,000. Preserve it unless a separately recorded, reproducible experiment supports a change.

## Stage 5 scorer reference

Initial held-out scorer baseline: F0.5=0.5483, precision=0.9030, recall=0.2510, threshold=12.0. Of 482 true links, 17 were not generated as candidates, 465 were scored, 121 scored ≥12, and 344 scored below 12. The full frozen test candidate volume is estimated at ~12.67B pairs and cannot feasibly be materialized. Submission #1 instead used fast indexed exact/strong rules and scored F0.5=0.228 on the leaderboard.

## Workstream 1 — Submission #1 diagnosis (Priority 1)

Reproduce the exact Submission #1 algorithm on held-out training data. Do not change its implementation initially. Measure candidate recall, precision, F0.5, zero-match behavior, singleton behavior, and multi-match behavior. Compare validation behavior with the 0.228 leaderboard result. Diagnose false positives/negatives, multilingual failures, missing-address failures, and ambiguous exact-name/address failures. Record split, seed, rules, parameters, runtime, and limitations in a new experiment directory. Preserve Submission #1 artifacts. **Owner:** teammate to be assigned.

## Workstream 2 — Rare-key candidate retrieval (Priority 2)

Design computationally feasible retrieval without materializing billions of candidates. Investigate rare name keys, rare address tokens, country-aware inverted indexes, exact/near-exact retrieval, posting-list caps, source-specific S2/S3 handling, multilingual/script-aware keys, and missing-address cases. Do not materialize ~12.67B frozen candidates. Alternatives must be separate, recorded experiments.

## Workstream 3 — Pair scoring (Priority 3)

Build a fast scorer over retrieved pairs. Investigate normalized exact name/address, abbreviation-normalized address, token overlap, character similarity, name/address agreement, source-specific behavior, and threshold selection using held-out S1-grouped validation. Keep the scorer simple and reproducible.

## Workstream 4 — Multilingual and missing-address recovery (Priority 4)

Investigate Latin-to-Indian-script names, missing target addresses, name order, and typo-like names. Do not use external data/API lookup. Validate any normalization or transliteration change experimentally.

## Workstream 5 — Integration and Submission #2 (Priority 5)

**Owner:** project lead. Integrate validated improvements, run the official validator, verify candidate-pair consistency, produce outputs, maintain submission count, record leaderboard results, and preserve exact experiment configuration. Submit only after evaluation and output validation.

## CURRENT RULES

1. Official challenge PDFs remain authoritative.
2. Do not use prohibited external data/API lookups.
3. Do not modify the original dataset.
4. Do not commit dataset/output/database files.
5. Preserve reproducibility.
6. Log every experiment.
7. Do not overwrite historical experiment results.
8. Do not silently change frozen Stage 4 decisions.
9. Never assume a leaderboard score corresponds to an offline experiment unless it used the same algorithm.
10. Record submission count and status.
