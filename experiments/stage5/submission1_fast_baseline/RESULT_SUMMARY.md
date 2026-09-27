# Submission #1 fast direct-match baseline — run summary

- **Run date:** 2026-09-26
- **Test S1 rows processed:** 1,732,544
- **Predicted target IDs:** 510,033 across 499,607 S1 rows
- **S1 prediction distribution:** 1,232,937 empty; 489,960 with one ID; 9,647 with multiple IDs
- **Benchmark:** 10,000 rows in 3.952 s = 2,530.45 rows/s; estimated lookup-only full pass ~11 min 24 s
- **Full inference:** 539.841 s = 8 min 59.8 s (3,209.36 rows/s)
- **Index build:** 9,574.332 s = 2 h 39 min 34 s; four posting tables contain 38,745,838 rows total; SQLite index 3,920,707,584 bytes
- **Combined build + benchmarked full inference wall time:** approximately 10,114 s = 2 h 48 min 34 s
- **Peak RSS:** 357,875,712 bytes during index build/benchmark; 22,970,368 bytes during full inference
- **Output sizes:** `matching_results.tsv` 30,138,267 bytes; `candidate_pairs.tsv` 30,138,269 bytes
- **Validation:** official validator PASS with both files (all formatting, required S1 coverage, duplicates, and match containment). A second validator run with `--check-ids` PASSed the matching output against all 9,969,589 test target IDs. Candidate IDs are copied from the matching output by the same implementation, so they have the same existence guarantee.
- **Candidate output scope:** each candidate row contains the accepted direct-match IDs only. This compact Submission #1 representation is not the full Stage 4 blocker candidate set. Stage 4 remains frozen and unchanged.

Reproduce with `python3 experiments/stage5/submission1_fast_baseline/fast_baseline.py --full` after the index has been built. To rebuild it, use `--full --rebuild-index`. The script uses only Python stdlib and the supplied normalization code; it installs no packages and does not access external data.
