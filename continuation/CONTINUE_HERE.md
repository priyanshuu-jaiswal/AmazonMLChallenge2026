# Continue Here — Amazon ML Challenge 2026

**Project:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Current date:** 2026-09-27  
**Project root:** `/Users/priyanshujaiswal/Documents/AmazonMLChallenge2026`  
**Current objective:** Diagnose Submission #1 and prepare Submission #2.

## Challenge and authoritative requirements

Source 1 is the reference set; Source 2 and Source 3 are candidate sets. Training ground truth supplies the known links. Optimize macro-averaged F0.5 (β=0.5), which weights precision more strongly than recall. Singletons are included. The official challenge PDFs and the supplied challenge package README are authoritative. Use only files in `student_resource/`; do not use external business/entity lookup, APIs, or augmentation. Treat country as open-set: the test includes France even though training has US and India. Preserve all original `student_resource/` files.

## Stage status

- Stage 1: COMPLETE
- Stage 2: COMPLETE
- Stage 3: COMPLETE
- Stage 4.1–4.7: COMPLETE; Stage 4 COMPLETE / FROZEN
- Stage 5: INCOMPLETE (Submission #1 direct-match baseline is complete; Submission #2 diagnosis and improvement remain)

See the project’s `PROJECT_STATE.md` and `EXPERIMENT_LOG.md` for historical records. Those records predate the Stage 5 work summarized here and contain stale statements that Stage 5 has not started; use the experiment artifacts under `experiments/stage5/` as the Stage 5 record and update project records deliberately when appropriate.

## Frozen Stage 4 design

Define **A = Prefix5 UNION Address2500**. The frozen operational generator is **A + HF-address/name**, adding pairs satisfying:

- same-country address token with document frequency 2,501–25,000, **AND**
- same-country name token with document frequency 1–25,000.

Do not change these definitions without explicitly recording a new experiment. Existing normalization, country constraints, key definitions, and combined S2+S3 frequency definitions remain as recorded in Stage 4 artifacts.

Stage 4.7 used a 1,500-S1 population-weighted validation sample with 5,156 true links (S2=2,505; S3=2,651):

| Strategy | Recalled links | Candidates | S2 candidates | S3 candidates |
|---|---:|---:|---:|---:|
| A | 4,959/5,156 (96.18%) | 10,906,369 | 5,253,117 | 5,653,252 |
| B | 5,096/5,156 (98.84%) | 36,380,324 | 17,796,817 | 18,583,507 |
| A + HF-address/name | 5,023/5,156 (97.42%) | 10,967,173 | 5,280,699 | 5,686,474 |
| A + name-token≤2,500 | 5,034/5,156 (97.63%) | 11,770,701 | 5,674,400 | 6,096,301 |

Stage 4 was frozen using A+HF-address/name. Full evidence: `experiments/stage4_7/report.md`, `results.json`, and `sample_ids.txt`.

## Stage 5 scorer baseline

Validation F0.5=**0.5483**, precision=**0.9030**, recall=**0.2510**, selected threshold=**12.0**. Of 482 true validation links, 17 were not generated; 465 were scored; 121 scored ≥12 and 344 scored below 12. True-link score median=8.17. Threshold 12 is the measured macro-F0.5 optimum for this scorer on this validation sample (not a global-optimality claim). Threshold 8: recall 53.3%, precision 9.7%; threshold 6: recall 79.0%, precision 3.2%. No feature-wiring bug was found; low recall is the measured behavior of the current simple scorer. Details are in `experiments/stage5/validation_summary.md`, `threshold_results.json`, and `scorer_diagnostics.json`.

## Current blocker and outputs

Full test candidate generation is too large to materialize. Strategy A alone extrapolates to about **12.60 billion pairs**, about **164 GB** as plain TSV, against roughly **18 GiB** available disk at the time this was assessed. Do not materialize the full set. `experiments/stage5/stream_infer.py` implements one-S1-at-a-time candidate generation and immediate scoring, but inspect it and its storage preflight before use. Intended flow: generate candidates → score immediately → retain accepted matches → discard rejected candidates.

Submission #1 is complete: the fast direct indexed baseline processed 1,732,544 S1s and predicted 510,033 IDs (1,232,937 zero-match rows, 489,960 one-match rows, 9,647 multi-match rows). The validator passed and the leaderboard score was F0.5=0.228. This score came from the fast baseline, not the full frozen Stage 4 + Stage 5 scorer. Outputs are local under `output/` and ignored by Git. The compact candidate file contains accepted IDs only, not the full frozen Stage 4 blocker set. Details are in `experiments/stage5/submission1_fast_baseline/`.

## Important project records and artifacts

Preserve/reference `AGENTS.md`, `PROJECT_STATE.md`, `EXPERIMENT_LOG.md`, `normalization/`, `experiments/stage4_3/` through `experiments/stage4_7/`, and `experiments/stage5/`. The core streaming script is `experiments/stage5/stream_infer.py`. Convenient copies of continuation scripts are under `continuation/scripts/`; the authoritative originals remain in the experiment directories.

## Start here

START HERE:

Read this file first, then PROJECT_STATE.md and EXPERIMENT_LOG.md.

The next task is NOT Stage 4 research.

The next task is Submission #2 diagnosis.

First reproduce the exact Submission #1 algorithm on held-out training data and diagnose its errors. Then design computationally feasible higher-recall retrieval/scoring. Do not score 12.6 billion candidates, materialize hundreds of GB, or change frozen Stage 4 methodology without recording a new experiment.

Current Stage 5 baseline:
F0.5 0.5483
Precision 0.9030
Recall 0.2510
Threshold 12.0

Submission #1 leaderboard score: 0.228.
