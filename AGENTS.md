# Amazon ML Challenge 2026 — Project Rules

## Challenge
Business Entity Resolution: for every Source 1 business, identify zero or more matching Source 2 and Source 3 records.

## Permanent rules
- Use only files supplied in `student_resource/`. Do not use external business/entity lookup, APIs, external data, or data augmentation.
- Preserve the original `student_resource/` files. Do not edit, move, rename, or delete them.
- Process large TSVs with streaming or chunked methods; do not load all multi-million-row sources into memory at once.
- Treat country as open-set: training has US and India; test also has France. Do not hard-code a training-only country list.
- Optimize the official macro-averaged F0.5 metric. Precision is weighted twice as strongly as recall; singletons are included.
- Final model must be licensed MIT or Apache 2.0 and have at most 8 billion parameters.

## Output contract
- Produce tab-separated `matching_results.tsv` and `candidate_pairs.tsv` in the final package's `output/` folder, with the exact headers and ID-list formats specified in `student_resource/README.md`.
- Include exactly one `matching_results.tsv` row for every test Source 1 ID; use an empty match list for no-match cases.
- Candidate pairs must contain every final match. Final IDs must exist in test Source 2 or Source 3; never emit duplicates, Source 1 self-matches, or duplicate Source 1 rows.
- Validate both files with the supplied `student_resource/utils/validate_submission.py` before submission.
- Final package must follow the README structure: outputs, reproducible code under `code/business_entity_resolution/`, pinned dependencies, and completed methodology documentation.

## Reproducibility and change control
- Record seeds, data split, methods, configuration, and results so each run can be reproduced.
- Do not silently change methodology. Explain material changes and decisions in both `PROJECT_STATE.md` and `EXPERIMENT_LOG.md`.
- Follow the checkpoints in `PROJECT_STATE.md`; do not advance until the relevant checkpoint passes.
