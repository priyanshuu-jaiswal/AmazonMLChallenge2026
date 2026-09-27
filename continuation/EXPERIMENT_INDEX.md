# Experiment Index

Authoritative experiment artifacts remain in the project’s `experiments/` directory. Continuation scripts are copied under `continuation/scripts/` for convenience.

- **`experiments/stage4_3/` — Reproducible held-out blocking validation.** Seeded 300-S1 balanced validation comparing Prefix5/address-cap strategies; includes reproducible runner and sample IDs.
- **`experiments/stage4_4/` — Targeted blocking recovery.** Inspects baseline misses and tests targeted address/name recovery rules on an independent sample.
- **`experiments/stage4_5/` — Multi-sample validation.** Repeats predeclared blocking strategies over three samples and records recall, volume, and diagnostics.
- **`experiments/stage4_6/` — Candidate-generator design research.** Measures posting frequencies, per-pass/per-source candidate volumes, overlap, and recall on a new sample.
- **`experiments/stage4_7/` — Final candidate-generation validation and freeze.** Population-weighted 1,500-S1 comparison, reproducibility records, `report.md`, `results.json`, and `sample_ids.txt`; selected frozen A+HF-address/name.
- **`experiments/stage5/` — First scorer, validation, and streaming inference.** Scorer/features/candidate generator, threshold and diagnostics artifacts, validation IDs/summary, and `stream_infer.py`. Baseline threshold 12 has F0.5 0.5483, precision 0.9030, recall 0.2510. Test inference has not run.
