"""Reproduce bounded Stage 3.3 normalization checks on supplied training data.

The deterministic sample is the first 128 nonempty ground-truth rows. Source
files are streamed only to retrieve these S1 and matched target IDs; only those
selected records are normalized or compared.
"""

from __future__ import annotations

import csv
import json
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

from normalize import (
    address_token_set,
    normalize_address,
    normalize_address_abbreviations,
    normalize_name,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "student_resource" / "dataset" / "train"
SAMPLE_ENTITIES = 128


def load_ground_truth_sample() -> dict[str, set[str]]:
    selected: dict[str, set[str]] = {}
    with (TRAIN / "train_ground_truth.tsv").open(
        encoding="utf-8", newline=""
    ) as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["matched_entity_ids"]:
                selected[row["source1_entity_id"]] = set(
                    row["matched_entity_ids"].split(",")
                )
                if len(selected) == SAMPLE_ENTITIES:
                    break
    if len(selected) != SAMPLE_ENTITIES:
        raise RuntimeError(f"Expected {SAMPLE_ENTITIES} sampled entities")
    return selected


def retrieve_selected_records(
    selected: dict[str, set[str]],
) -> dict[str, dict[str, str]]:
    wanted = set(selected)
    target_ids = set().union(*selected.values())
    records: dict[str, dict[str, str]] = {}
    files = (
        ("train_source1.tsv", wanted),
        ("train_source2.tsv", {key for key in target_ids if key.startswith("S2-")}),
        ("train_source3.tsv", {key for key in target_ids if key.startswith("S3-")}),
    )
    for filename, ids in files:
        if not ids:
            continue
        with (TRAIN / filename).open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                entity_id = row["entity_id"]
                if entity_id in ids:
                    records[entity_id] = row
                    if len({key for key in records if key in ids}) == len(ids):
                        break
    missing = (wanted | target_ids) - records.keys()
    if missing:
        raise RuntimeError(f"Could not retrieve {len(missing)} sampled IDs")
    return records


def record_pair(a: dict[str, str], b: dict[str, str]) -> dict[str, object]:
    left_tokens = address_token_set(a["business_address"])
    right_tokens = address_token_set(b["business_address"])
    return {
        "s1_id": a["entity_id"],
        "s1_country": a["country"],
        "s1_name_raw": a["business_name"],
        "target_id": b["entity_id"],
        "target_country": b["country"],
        "target_name_raw": b["business_name"],
        "s1_name_normalized": normalize_name(a["business_name"]),
        "target_name_normalized": normalize_name(b["business_name"]),
        "s1_address_raw": a["business_address"],
        "target_address_raw": b["business_address"],
        "s1_address_normalized": normalize_address(a["business_address"]),
        "target_address_normalized": normalize_address(b["business_address"]),
        "s1_address_abbreviations": normalize_address_abbreviations(
            a["business_address"]
        ),
        "target_address_abbreviations": normalize_address_abbreviations(
            b["business_address"]
        ),
        "address_token_overlap": (
            sorted(left_tokens & right_tokens)
            if left_tokens is not None and right_tokens is not None
            else None
        ),
        "address_token_jaccard": jaccard(a, b),
    }


def same_nonempty(left: str | None, right: str | None) -> bool:
    return bool(left) and bool(right) and left == right


def jaccard(a: dict[str, str], b: dict[str, str]) -> float | None:
    left = address_token_set(a["business_address"])
    right = address_token_set(b["business_address"])
    if left is None or right is None:
        return None
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def script_present(text: str, low: int, high: int) -> bool:
    return any(low <= ord(char) <= high for char in text)


def emit_examples(title: str, pairs: Iterable[tuple[dict[str, str], dict[str, str]]], limit: int = 3) -> None:
    print(title)
    count = 0
    for a, b in pairs:
        print(json.dumps(record_pair(a, b), ensure_ascii=False))
        count += 1
        if count >= limit:
            break
    if count == 0:
        print("  none in this deterministic sample")


def main() -> None:
    selected = load_ground_truth_sample()
    records = retrieve_selected_records(selected)
    target_ids = set().union(*selected.values())
    matches = [
        (records[s1_id], records[target_id])
        for s1_id, ids in selected.items()
        for target_id in sorted(ids)
    ]
    nonmatches = [
        (records[s1_id], records[target_id])
        for s1_id, ids in selected.items()
        for target_id in sorted(target_ids - ids)
    ]

    raw_name = lambda a, b: a["business_name"] == b["business_name"]
    norm_name = lambda a, b: same_nonempty(
        normalize_name(a["business_name"]), normalize_name(b["business_name"])
    )
    raw_address = lambda a, b: same_nonempty(
        a["business_address"], b["business_address"]
    )
    norm_address = lambda a, b: same_nonempty(
        normalize_address(a["business_address"]),
        normalize_address(b["business_address"]),
    )
    abbr_address = lambda a, b: same_nonempty(
        normalize_address_abbreviations(a["business_address"]),
        normalize_address_abbreviations(b["business_address"]),
    )
    token_equal = lambda a, b: (
        address_token_set(a["business_address"]) is not None
        and address_token_set(a["business_address"])
        == address_token_set(b["business_address"])
    )

    scores = [score for a, b in matches if (score := jaccard(a, b)) is not None]
    sorted_scores = sorted(scores)

    def quantile(fraction: float) -> float | None:
        if not sorted_scores:
            return None
        return sorted_scores[round((len(sorted_scores) - 1) * fraction)]

    print(f"sample_s1_rows={len(selected)}")
    print(f"known_match_pairs={len(matches)}")
    print(f"unique_sampled_target_records={len(target_ids)}")
    print(f"constructed_labeled_nonmatch_pairs={len(nonmatches)}")
    print(f"sample_s1_countries={dict(Counter(records[x]['country'] for x in selected))}")
    print(
        "sample_target_countries="
        f"{dict(Counter(records[x]['country'] for x in target_ids))}"
    )
    print(
        "match_exact_counts="
        f"name_raw:{sum(raw_name(a,b) for a,b in matches)},"
        f"name_normalized:{sum(norm_name(a,b) for a,b in matches)},"
        f"address_raw:{sum(raw_address(a,b) for a,b in matches)},"
        f"address_normalized:{sum(norm_address(a,b) for a,b in matches)},"
        f"address_abbreviations:{sum(abbr_address(a,b) for a,b in matches)},"
        f"address_token_sets:{sum(token_equal(a,b) for a,b in matches)}"
    )
    print(
        "match_address_jaccard="
        f"n:{len(scores)},min:{min(scores) if scores else None},"
        f"p10:{quantile(.10)},median:{statistics.median(scores) if scores else None},"
        f"p90:{quantile(.90)},max:{max(scores) if scores else None},"
        f"ge_0.5:{sum(x >= .5 for x in scores)},ge_0.8:{sum(x >= .8 for x in scores)}"
    )
    nonmatch_scores = [
        score for a, b in nonmatches if (score := jaccard(a, b)) is not None
    ]
    print(
        "nonmatch_collision_counts="
        f"name_normalized:{sum(norm_name(a,b) for a,b in nonmatches)},"
        f"address_normalized:{sum(norm_address(a,b) for a,b in nonmatches)},"
        f"address_abbreviations:{sum(abbr_address(a,b) for a,b in nonmatches)},"
        f"address_token_sets:{sum(token_equal(a,b) for a,b in nonmatches)},"
        f"jaccard_ge_0.5:{sum(x >= .5 for x in nonmatch_scores)},"
        f"jaccard_ge_0.8:{sum(x >= .8 for x in nonmatch_scores)},"
        f"max_jaccard:{max(nonmatch_scores) if nonmatch_scores else None}"
    )

    missing_matches = [
        (a, b) for a, b in matches if not a["business_address"] or not b["business_address"]
    ]
    print(f"known_matches_with_an_empty_address={len(missing_matches)}")
    emit_examples(
        "name_normalization_helped",
        ((a, b) for a, b in matches if not raw_name(a, b) and norm_name(a, b)),
    )
    emit_examples(
        "name_normalization_did_not_make_equal",
        ((a, b) for a, b in matches if not norm_name(a, b)),
    )
    emit_examples(
        "address_normalization_helped",
        ((a, b) for a, b in matches if not raw_address(a, b) and norm_address(a, b)),
    )
    emit_examples(
        "abbreviation_representation_added_exact_equality",
        ((a, b) for a, b in matches if not norm_address(a, b) and abbr_address(a, b)),
    )
    ranked_matches = sorted(
        ((jaccard(a, b), a, b) for a, b in matches if jaccard(a, b) is not None),
        key=lambda item: item[0],
        reverse=True,
    )
    emit_examples(
        "high_token_overlap_match_examples",
        ((a, b) for score, a, b in ranked_matches if score >= .5 and not norm_address(a, b)),
    )
    ranked_nonmatches = sorted(
        ((jaccard(a, b), a, b) for a, b in nonmatches if jaccard(a, b) is not None),
        key=lambda item: item[0],
        reverse=True,
    )
    emit_examples(
        "highest_overlap_labeled_nonmatches",
        ((a, b) for _, a, b in ranked_nonmatches),
    )
    emit_examples("empty_address_match_examples", missing_matches)

    scripts = {
        "Devanagari": (0x0900, 0x097F),
        "Tamil": (0x0B80, 0x0BFF),
        "Kannada": (0x0C80, 0x0CFF),
    }
    for script, (low, high) in scripts.items():
        candidates = [
            (a, b) for a, b in matches
            if script_present(a["business_name"] + a["business_address"], low, high)
            or script_present(b["business_name"] + b["business_address"], low, high)
        ]
        print(f"script_examples[{script}]={len(candidates)}")
        emit_examples(f"{script}_matched_examples", candidates, limit=2)

    other_script = [
        (a, b) for a, b in matches
        if any(
            ord(char) > 0x024F
            and not any(low <= ord(char) <= high for low, high in scripts.values())
            for char in a["business_name"] + a["business_address"] + b["business_name"] + b["business_address"]
        )
    ]
    print(f"other_non_latin_script_matches={len(other_script)}")
    emit_examples("other_script_examples", other_script, limit=2)

    # Per-sample idempotence check in addition to the focused unit tests.
    strings = [
        value
        for row in records.values()
        for value in (row["business_name"], row["business_address"])
    ]
    assert all(
        normalize_name(normalize_name(value)) == normalize_name(value)
        and normalize_address(normalize_address(value)) == normalize_address(value)
        and normalize_address_abbreviations(
            normalize_address_abbreviations(value)
        ) == normalize_address_abbreviations(value)
        for value in strings
    )
    print(f"sample_string_idempotence=PASS ({len(strings)} values)")


if __name__ == "__main__":
    main()
