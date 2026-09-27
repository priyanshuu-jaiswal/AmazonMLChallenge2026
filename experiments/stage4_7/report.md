# Stage 4.7 — Final Candidate-Generation Validation

## Sample

- 1,500 S1 entities (5× the previous 300-S1 samples), selected by population-proportional quotas over match class × training country; seed 20261010.
- Quotas: India 600 / US 900; zero-match 83, one-match 81, multi-match 1,336. Exact stratum populations and quotas are recorded in `results.json`.
- 5,156 true links: S2=2,505; S3=2,651.
- SHA-256 of ordered IDs: `0cbae83ea46220e4efbe161c984ac45e0202ea8d153c4b835c55a93b200da9d5`.
- Disjoint from saved Stage 4.3, 4.4, 4.5, and 4.6 IDs. Exact Stage 4.1/4.2 IDs are unavailable.

## Strategy results

Candidate totals are deduplicated S1-target pairs; S2+S3 counts sum to the total. Quantiles are per-S1.

| Rule | Recovered / true (recall) | S2 links | S3 links | Total pairs (S2 + S3) | Mean/S1 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A: Prefix5 ∪ Address2500 | 4,959/5,156 (96.18%) | 2,420/2,505 (96.61%) | 2,539/2,651 (95.78%) | 10,906,369 (5,253,117 + 5,653,252) | 7,270.91 | 21,547 | 34,535 | 64,105 |
| B: Prefix5 ∪ Address25000 | 5,096/5,156 (98.84%) | 2,480/2,505 (99.00%) | 2,616/2,651 (98.68%) | 36,380,324 (17,796,817 + 18,583,507) | 24,253.55 | 55,510 | 75,313 | 127,644 |
| A+HF-address/name | 5,023/5,156 (97.42%) | 2,442/2,505 (97.49%) | 2,581/2,651 (97.36%) | 10,967,173 (5,280,699 + 5,686,474) | 7,311.45 | 21,557 | 34,648 | 64,105 |
| A+name-token df≤2,500 | 5,034/5,156 (97.63%) | 2,447/2,505 (97.68%) | 2,587/2,651 (97.59%) | 11,770,701 (5,674,400 + 6,096,301) | 7,847.13 | 21,760 | 34,535 | 64,105 |

## Zero-match tail

There are 83 zero-match S1s. Mean / median / P95 / P99 / max candidate counts were:

- A: 6,798 / 3,426 / 22,423 / 30,740 / 30,740.
- B: 24,132 / 23,093 / 59,890 / 72,551 / 72,551.
- A+HF-address/name: 6,835 / 3,426 / 22,530 / 30,759 / 30,759.
- A+name-token df≤2,500: 7,377 / 4,980 / 22,423 / 32,250 / 32,250.

All zero-match bins and totals are in `results.json`.

## Established failure slices

Slices overlap and follow Stage 4.5 definitions: script sets differ; target address is empty; normalized name token sets match while normalized strings differ; or normalized strings differ with SequenceMatcher ratio ≥0.72.

| Slice (true links) | A | B | A+HF-address/name | A+name-token≤2,500 |
|---|---:|---:|---:|---:|
| Multilingual/script (369) | 299 (81.03%) | 351 (95.12%) | 299 (81.03%) | 299 (81.03%) |
| Missing target address (217) | 205 (94.47%) | 205 (94.47%) | 205 (94.47%) | 217 (100%) |
| Name-order variation (308) | 291 (94.48%) | 304 (98.70%) | 304 (98.70%) | 306 (99.35%) |
| Typo-like name variation (2,831) | 2,760 (97.49%) | 2,809 (99.22%) | 2,802 (98.98%) | 2,807 (99.15%) |

## Reproducibility and decision

The experiment was run from scratch twice. Sample IDs/hash, recall, total and source candidate counts, per-S1 distributions, and failure-slice results matched exactly. See `reproducibility.json`.

Tradeoffs versus A: HF-address/name adds 64 recovered links for 60,804 pairs; name-token adds 75 for 864,332 pairs; B adds 137 for 25,473,955 pairs. No rule is called “best.” Based on the repeated Stage 4.5–4.7 evidence, Stage 4 is frozen for Stage 5 using the existing A+HF-address/name definition: it raises recall to 97.42% for 0.56% more pairs than A, while B's additional recall comes with much higher volume. This is an operational freeze for Stage 5, not a claim about final precision or F0.5.

Stage 4.7 did not modify `student_resource/`, create candidate-pair files, or start Stage 5. France is not represented in training; Stage 4.1/4.2 exact IDs are unavailable.
