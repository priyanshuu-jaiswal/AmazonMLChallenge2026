# Amazon ML Challenge 2026 — Team Handoff

## Project

Business Entity Resolution: for every Source 1 business, identify zero or more corresponding Source 2 and Source 3 records.

## Current status

- Stage 1: COMPLETE
- Stage 2: COMPLETE
- Stage 3: COMPLETE
- Stage 4: COMPLETE / FROZEN
- Stage 5: INCOMPLETE
- Submission #1: completed; leaderboard F0.5 = **0.228**

The 0.228 score came from the fast Submission #1 direct-match baseline, not the full frozen Stage 4 + Stage 5 scorer. The local output passed the supplied validator. Stage 5 still needs held-out reproduction and diagnosis before Submission #2.

## Dataset setup

The official dataset is intentionally not stored in Git. Obtain the official challenge files and place them locally under:

```text
student_resource/
├── dataset/
│   ├── train/
│   └── test/
├── utils/
└── ...
```

The challenge files are required locally. Do not redistribute or copy the large dataset into Git. Do not modify `student_resource/dataset/`.

## Frozen Stage 4 design

**A = Prefix5 UNION Address2500.** The frozen operational strategy is **A + HF-address/name**: same-country address token df 2,501–25,000 **AND** same-country name token df 1–25,000.

Stage 4.7 used 1,500 S1s and 5,156 true links:

| Strategy | Recall | Candidates |
|---|---:|---:|
| A | 96.18% (4,959/5,156) | 10,906,369 |
| A + HF-address/name (frozen) | 97.42% (5,023/5,156) | 10,967,173 |
| B | 98.84% (5,096/5,156) | 36,380,324 |

Do not silently change these definitions. Record any proposed change as a separate experiment. See `experiments/stage4_7/` for the validation evidence.

## Stage 5 scorer and compute limit

The held-out training validation scorer baseline measured F0.5=0.5483, precision=0.9030, recall=0.2510, threshold=12.0. It was not the source of Submission #1's leaderboard score.

The frozen generator is estimated to produce about 12.6 billion test candidates. At the measured ~19,600 pair scores/second, scoring the full set would take about 178 hours. Do not launch that job or materialize hundreds of GB of candidates. Stage 4 remains frozen; pursue a separately documented computationally feasible retrieval approach.

## Submission #1

The fast direct indexed exact/strong matching baseline processed 1,732,544 S1 rows and predicted 510,033 target IDs:

- 1,232,937 S1 rows with zero matches
- 489,960 S1 rows with one match
- 9,647 S1 rows with multiple matches

The supplied validator passed. The leaderboard score was F0.5=0.228. This is the first real leaderboard baseline, not the final approach. Its compact `candidate_pairs.tsv` is an accepted-ID representation for this submission and is not the complete frozen Stage 4 blocker set. See `experiments/stage5/submission1_fast_baseline/` for code, rules, benchmark, and results.

## Immediate next task: Submission #2

1. Reproduce the exact Submission #1 rules on held-out training data.
2. Measure F0.5, precision, recall, false positives, and false negatives.
3. Diagnose multilingual failures, missing-address failures, and ambiguous exact-name/address cases.
4. Develop computationally feasible higher-recall retrieval and scoring.
5. Validate Submission #2, then submit it to the leaderboard.

Never score the 12.6B candidate set or materialize hundreds of GB. Do not modify `student_resource/`, use external data/APIs, install unnecessary packages, delete historical experiments, or declare a best model before validation.

## History and control files

Preserve `experiments/stage4_3/`, `stage4_4/`, `stage4_5/`, `stage4_6/`, `stage4_7/`, and `stage5/`. `AGENTS.md`, `PROJECT_STATE.md`, and `EXPERIMENT_LOG.md` are project-control records. Continuation notes for fresh sessions are under `continuation/`.
