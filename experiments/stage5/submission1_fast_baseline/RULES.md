# Submission #1 direct-match baseline rules

This is a submission-only, storage-constrained baseline. It does not replace, revise, or claim to implement the frozen Stage 4 candidate-generation methodology.

All keys use the existing `normalization/normalize.py` functions and require exact country equality; country values are not enumerated, so France is handled like any other value.

1. Accept all target records with exact country + normalized name + normalized address.
2. Also accept all target records with exact country + normalized name + abbreviation-normalized address (the existing approved abbreviation mapping).
3. Only when neither exact pair rule yielded a result, accept a normalized-name-only posting if exactly one target record across test Source 2 and Source 3 has that country/name key.
4. Preserve every ID for duplicate exact pair keys; deduplicate the final union. Empty name/address values are never used as exact pair evidence. Empty output lists are written as empty TSV fields.

`candidate_pairs.tsv` for this baseline contains precisely the IDs accepted by these rules, so it contains every final match. It is a compact Submission #1 representation, not the complete candidate set from the frozen Stage 4 blocker.
