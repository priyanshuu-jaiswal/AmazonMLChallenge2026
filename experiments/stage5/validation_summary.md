# Stage 5 validation baseline

## Validation design

- 300 held-out S1 entities; 50 in each match-count class (zero, one, multi) × country (India, US).
- Entity-level deterministic split: SHA256 of `20261011`, NUL, and S1 ID; first digest byte modulo 5 equal to zero enters the held-out pool. Validation sampling uses deterministic bottom-k SHA256 rank with seed `20261012` within each stratum.
- Ordered validation-ID SHA256: `2218951d787c79fde254740c2c73e7b9375965ed43b8854d11c7c6c0ab56eb4`.
- The sample contains 482 true links (229 S2, 253 S3). No score fitting was performed; 20% of S1s were reserved as the held-out pool and the other 80% designated development/training IDs.

## Scorer and threshold

The implementation uses the frozen Stage 4.7 A+HF_NAME candidate rule and the existing Stage 3 normalization / Stage 4 key functions. The weighted score is deterministic and documented in `scorer.py`; its complete feature weights and the full threshold sweep are in `threshold_results.json`.

The selected sweep threshold is 12.0 (highest macro F0.5, tie-breaking toward the higher threshold): macro F0.5 0.5483, micro precision 0.9030, micro recall 0.2510. It predicted 134 links: 121 true positives, 13 false positives, and 361 false negatives. Among 100 zero-match S1s, 96 had no prediction. Candidate generation covered 465/482 true links (96.47%) before scoring. This is a baseline result, not a claim of global optimality.

## Scorer diagnosis and compact sweep

The validation was reproduced on the saved 300 IDs. Of 482 true links, 465 were candidate-generated and scored; 17 were absent from the candidate set. Of the 465 scored true links, 121 scored at least 12 and 344 scored below 12. True-link score median was 8.172 (mean 8.984; range 2.735–22.0). False-candidate scores (2,344,149 pairs) had median 3.268, p95 4.894, p99 5.659, and maximum 13.611. See `scorer_diagnostics.json` and `true_link_score_diagnostics.tsv` for the complete score distributions and every scored ground-truth pair with feature values.

The compact sweep confirms the threshold tradeoff: at 4, recall 0.932 but precision 0.00093 and macro F0.5 0.0434; at 6, recall 0.790, precision 0.0319 and macro F0.5 0.2587; at 8, recall 0.533, precision 0.0968 and macro F0.5 0.4258; at 12, recall 0.251, precision 0.9030 and macro F0.5 0.5483. The score implementation is not reversed or mis-scaled: all similarity features are 0–1, exact indicators are 0/1, the weighted sum increases with agreement, and exact-match bonuses work. The low recall at 12 is caused by selecting the precision-heavy macro-F0.5 optimum on this sample, not a feature wiring defect. The only measured threshold selected by the stated macro-F0.5 objective remains 12; lower thresholds recover more links at very poor precision. No arbitrary alternate baseline was substituted.

Among the 465 scored true links, name exactness fired for 86, conservative address exactness for 42, approved-abbreviation address exactness for 48, and at least one address was missing for 20. All 12 feature fields were present; all 465 candidate true links had nonzero name character similarity and country equality. This verifies that exact bonuses and similarity features are wired into the score.

## Submission generation status and storage limit


No test predictions or candidate-pair output were written, and the official validator was not run. The supplied test S1 file has 1,732,544 data rows. Stage 4.7 observed 7,311.45 candidate pairs per S1 on its deterministic 1,500-S1 training sample. A simple scale estimate is therefore about 12.66 billion test pairs; this is an extrapolation, not a measured test candidate count. The official candidate file stores a comma-separated ID list per S1 row; with 12-byte IDs this is roughly 13 bytes per candidate, or about 165 GB at that extrapolated volume, plus headers and row IDs. This corrects the earlier overestimate that counted one full row per pair. The workspace had only about 18 GiB free at inspection. The full required candidate output still cannot safely be generated in the available storage, so this run stops before writing partial or misleading submission files.

The official validator requires exact TSV headers `source1_entity_id` / `matched_entity_ids` and `source1_entity_id` / `candidate_entity_ids`; it checks one row per test S1, no duplicate IDs within lists, only S2/S3 prefixes, and prediction IDs as a subset of listed candidates. Candidate input is optional to the validator (absence is a warning, not a failure), but the challenge README and final package require `candidate_pairs.tsv` and define it as the complete set fed to inference. The validator opens plain text paths; gzip input is not accepted. A streaming writer controls RAM but does not remove the plain-text disk requirement. A compressed archive may compress the final package but cannot replace the plain `.tsv` during the required local validator run. The validator parses the candidate TSV into an in-memory dictionary of sets, so its documented “more memory” cost grows with every candidate ID; at billions of IDs, this also makes official validation impractical on ordinary workstation memory.

## Next action

Provide a workspace/output volume with sufficient free capacity for the complete candidate-pair TSV (or authorize a validator-compatible storage destination with that capacity), then generate both complete outputs and run the official validator. Stage 5 is not complete and project state / experiment log were not advanced.

Validation runtime: 745.185 seconds; diagnostic reproduction runtime: 687.692 seconds. Test candidate count, predicted-match count on test, and official validator result: not measured / not run.
