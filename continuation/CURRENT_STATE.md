# Current State (2026-09-26)

**Project:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Root:** `/Users/priyanshujaiswal/Documents/AmazonMLChallenge2026`  
**Goal:** Diagnose Submission #1 and prepare Submission #2.

Sources: Source 1 is the reference; Source 2 and Source 3 are candidates. Use supplied training labels and optimize macro F0.5 (β=0.5; precision weighted more than recall; singletons included). Official PDFs and package README govern. No external lookup/augmentation; preserve `student_resource/`; country is open-set (test includes France).

Stages 1–3 complete. Stage 4.1–4.7 complete and frozen. Frozen A=Prefix5 ∪ Address2500; operational generator=A plus same-country address token df 2,501–25,000 AND same-country name token df 1–25,000. Stage 4.7 (1,500 S1 / 5,156 links): A 4,959 (96.18%) / 10.91M pairs; B 5,096 (98.84%) / 36.38M; frozen A+HF-address/name 5,023 (97.42%) / 10.97M; A+name-token≤2,500 5,034 (97.63%) / 11.77M. Do not alter without a recorded experiment.

Stage 5 is incomplete. Existing scorer validation: threshold 12, F0.5 0.5483, precision 0.9030, recall 0.2510; 17/482 true links were outside candidates, 121/465 scored links cleared threshold. No wiring bug found. Artifacts: `experiments/stage5/`.

Submission #1 is complete: the fast indexed direct-match baseline processed 1,732,544 S1 rows, predicted 510,033 IDs (1,232,937 zero-match rows; 489,960 one-match rows; 9,647 multi-match rows), passed the validator, and scored F0.5=0.228 on the leaderboard. This was not the full frozen Stage 4 + Stage 5 scorer. Outputs are local under ignored `output/`; details are in `experiments/stage5/submission1_fast_baseline/`.

Next: reproduce the exact Submission #1 rules on held-out training data; measure metrics and diagnose false positives/negatives, multilingual failures, missing-address failures, and ambiguous-key cases; then develop feasible higher-recall retrieval and scoring for Submission #2. Never score 12.6B candidates or materialize hundreds of GB. Read `CONTINUE_HERE.md`, root `PROJECT_STATE.md`, `EXPERIMENT_LOG.md`, `TEAM_README.md`, and `TEAM_TASKS.md`.
