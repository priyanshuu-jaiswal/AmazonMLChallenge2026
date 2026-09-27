#!/usr/bin/env python3
"""Reproducible Stage 4.3 held-out blocking validation (research only).

Run from the project root:
  python3 experiments/stage4_3/reproducible_validation.py

Uses only supplied training TSVs and the approved Stage 3 normalization
functions. Builds only postings for sample-query keys; never writes candidates.
"""
from __future__ import annotations

import csv
import difflib
import hashlib
import json
import os
import random
import re
import sqlite3
import statistics
import sys
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "student_resource" / "dataset" / "train"
OUT = ROOT / "experiments" / "stage4_3" / "latest_results.json"
SEED = 20260928
SAMPLE_PER_STRATUM = 50
HASH_PARTITION = 4  # held-out universe is IDs whose SHA-256 first byte is < 64
CAPS = (2500, 25000)
SOURCES = ("train_source2.tsv", "train_source3.tsv")

sys.path.insert(0, str(ROOT))
from normalization.normalize import (  # noqa: E402
    normalize_address,
    normalize_address_abbreviations,
    normalize_name,
)


def sha256_lines(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(ids)).encode("utf-8")).hexdigest()


def qtile(values: list[int], p: float) -> int | float:
    if not values:
        return 0
    s = sorted(values)
    return s[max(0, min(len(s) - 1, int(p * len(s) + 0.999999999) - 1))]


def dist(values: list[int]) -> dict:
    return {
        "n": len(values), "min": min(values) if values else 0,
        "mean": statistics.fmean(values) if values else 0,
        "median": statistics.median(values) if values else 0,
        "p90": qtile(values, .90), "p95": qtile(values, .95),
        "p99": qtile(values, .99), "max": max(values) if values else 0,
        "total": sum(values),
    }


def rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t")


def clean_tokens(value: str | None) -> set[str]:
    return set((value or "").split())


def partitioned(sid: str) -> bool:
    # Exact deterministic fallback holdout partition, used because historic IDs
    # are not present in records and cannot be excluded directly.
    return hashlib.sha256(sid.encode("utf-8")).digest()[0] < 64


def build_sample(db: sqlite3.Connection) -> tuple[list[dict], dict]:
    cur = db.cursor()
    cur.executescript("DROP TABLE IF EXISTS s1; DROP TABLE IF EXISTS gt;")
    cur.execute("CREATE TABLE s1(id TEXT PRIMARY KEY,country TEXT,name TEXT)")
    cur.execute("CREATE TABLE gt(id TEXT PRIMARY KEY,links TEXT NOT NULL, n INTEGER NOT NULL)")
    cur.execute("CREATE INDEX gt_n ON gt(n)")
    p = DATA / "train_source1.tsv"
    for i, row in enumerate(rows(p), 1):
        cur.execute("INSERT INTO s1 VALUES(?,?,?)", (row["entity_id"], row["country"], row["business_name"]))
        if i % 100000 == 0:
            db.commit()
    db.commit()
    for i, row in enumerate(rows(DATA / "train_ground_truth.tsv"), 1):
        ids = row["matched_entity_ids"].strip()
        links = [x.strip() for x in ids.split(",") if x.strip()] if ids else []
        cur.execute("INSERT INTO gt VALUES(?,?,?)", (row["source1_entity_id"], json.dumps(links), len(links)))
        if i % 100000 == 0:
            db.commit()
    db.commit()
    # Candidate IDs are externally sorted before random.sample, one stratum at a time.
    rng = random.Random(SEED)
    selected: list[tuple[str, str, int, str]] = []
    exclusions = Counter()
    for nclass, low, high in (("zero", 0, 0), ("one", 1, 1), ("multi", 2, 1000000)):
        for country in ("India", "US"):
            ids = [r[0] for r in cur.execute(
                "SELECT s.id FROM s1 s JOIN gt g ON g.id=s.id WHERE s.country=? AND g.n BETWEEN ? AND ? ORDER BY s.id",
                (country, low, high)) if partitioned(r[0])]
            if len(ids) < SAMPLE_PER_STRATUM:
                raise RuntimeError(f"Insufficient held-out stratum {nclass}/{country}: {len(ids)}")
            picked = rng.sample(ids, SAMPLE_PER_STRATUM)
            selected.extend((sid, country, low if nclass != "multi" else 2, nclass) for sid in picked)
            exclusions[f"not_hash_partition_{nclass}_{country}"] = "all IDs outside sha256(id)[0] < 64 excluded"
    sample = []
    for sid, country, _, cls in selected:
        linkjson, n = cur.execute("SELECT links,n FROM gt WHERE id=?", (sid,)).fetchone()
        name = cur.execute("SELECT name FROM s1 WHERE id=?", (sid,)).fetchone()[0]
        sample.append({"id": sid, "country": country, "class": cls, "name": name,
                       "links": json.loads(linkjson), "match_count": n})
    class_order = {"zero": 0, "one": 1, "multi": 2}
    country_order = {"India": 0, "US": 1}
    sample.sort(key=lambda x: (class_order[x["class"]], country_order[x["country"]], x["id"]))
    ids = [x["id"] for x in sample]
    # Reconstruct the selection independently from the staged source rows and
    # verify the exact ordered ID/class/country sequence before any evaluation.
    replay_rng = random.Random(SEED)
    replay = []
    for nclass, low, high in (("zero", 0, 0), ("one", 1, 1), ("multi", 2, 1000000)):
        for country in ("India", "US"):
            pool = [r[0] for r in cur.execute(
                "SELECT s.id FROM s1 s JOIN gt g ON g.id=s.id WHERE s.country=? AND g.n BETWEEN ? AND ? ORDER BY s.id",
                (country, low, high)) if partitioned(r[0])]
            replay.extend((sid, country, nclass) for sid in replay_rng.sample(pool, SAMPLE_PER_STRATUM))
    replay.sort(key=lambda x: (class_order[x[2]], country_order[x[1]], x[0]))
    expected = [(x["id"], x["country"], x["class"]) for x in sample]
    if replay != expected:
        raise AssertionError("independent sample-generation replay differed")
    return sample, {
        "source_files": ["student_resource/dataset/train/train_ground_truth.tsv", "student_resource/dataset/train/train_source1.tsv"],
        "filtering": "Use every ground-truth row whose S1 ID has a matching Source 1 row; include only exact Source 1 country values India or US; exclude rows outside the deterministic hash partition described below.",
        "match_classification": "Parse matched_entity_ids as comma-separated nonempty trimmed IDs; count 0=zero, exactly 1=one, >=2=multi.",
        "country_filter": "Exact string equality to India or US; France or any other value is excluded.",
        "seed": SEED, "rng": "Python random.Random (stdlib)",
        "population_order": "eligible IDs sorted lexicographically within each exact match-count × country stratum before random.sample",
        "sampling": "without replacement; 50 per zero/one/multi × India/US stratum; strata sampled in zero, one, multi then India, US order using one RNG instance",
        "hash_partition": "eligible only when SHA-256(UTF-8 S1 ID) first digest byte is <64; all other IDs excluded",
        "insufficient_strata": "abort before evaluation; do not substitute or reduce quotas",
        "final_order": "match class zero, one, multi; then country India, US; then S1 ID lexicographically",
        "sample_hash": sha256_lines(ids),
        "sample_hash_input": "UTF-8 final ordered S1 IDs joined by one LF byte, no trailing LF",
        "sample_generation_replay": "passed: fresh RNG instance and a second independently queried sorted pool reproduced identical ordered (S1 ID, country, match class) tuples",
        "historical_exclusion_limit": "Exact Stage 4.1/4.2/old 4.3 IDs are not logged, and old sampling code/ID lists are absent. This hash-partition holdout is the deterministic fallback; disjointness from historical samples cannot be proven.",
    }


def norm_fields(sample: list[dict]) -> None:
    for x in sample:
        x["nn"] = normalize_name(x["name"]) or ""
        x["prefix"] = x["nn"][:5] if len(x["nn"]) >= 5 else ""
        ca = normalize_address(x.get("address")) or ""
        aa = normalize_address_abbreviations(x.get("address")) or ""
        x["address_keys"] = clean_tokens(ca) | clean_tokens(aa)


def fetch_sample_data(sample: list[dict]) -> None:
    # Add Source 1 address from the supplied file; sample IDs are only 300.
    need = {x["id"]: x for x in sample}
    for r in rows(DATA / "train_source1.tsv"):
        x = need.get(r["entity_id"])
        if x is not None:
            x["address"] = r["business_address"]
            if len(need) == sum("address" in z for z in sample):
                break
    norm_fields(sample)


def keys_for_target(row: dict) -> tuple[tuple[str, str], set[tuple[str, str]]]:
    country = row["country"]
    nn = normalize_name(row["business_name"]) or ""
    pfx = (country, nn[:5]) if len(nn) >= 5 else (country, "")
    a = (clean_tokens(normalize_address(row["business_address"])) |
         clean_tokens(normalize_address_abbreviations(row["business_address"])))
    return pfx, {(country, t) for t in a}


def classify_miss(s1: dict, target: dict, addr_df: dict) -> list[str]:
    a, b = s1["nn"], normalize_name(target["name"]) or ""
    ta, tb = set(a.split()), set(b.split())
    labels = []
    if not target["business_address"].strip(): labels.append("missing target address")
    scripts = lambda s: {unicodedata.name(c, "").split(" ")[0] for c in s if c.isalpha()}
    if scripts(a) != scripts(b) and (scripts(a) - {"LATIN"}) != (scripts(b) - {"LATIN"}):
        labels.append("multilingual/script difference")
    if len(ta) > 1 and ta == tb and a != b: labels.append("name word order")
    if a and b and (a.startswith(b[:5]) or b.startswith(a[:5])) and a != b: labels.append("name abbreviation")
    ratio = difflib.SequenceMatcher(None, a, b).ratio() if a and b else 0
    if ratio >= .72 and a != b: labels.append("name typo")
    if not labels and (not a or len(a) < 5): labels.append("rare/noisy name")
    source_addr = clean_tokens(normalize_address(s1["address"]))
    target_addr = clean_tokens(normalize_address(target["business_address"]))
    expanded = clean_tokens(normalize_address_abbreviations(s1["address"]))
    expanded_t = clean_tokens(normalize_address_abbreviations(target["business_address"]))
    shared = (source_addr | expanded) & (target_addr | expanded_t)
    aliases = ({"st", "street"}, {"dr", "drive"}, {"ave", "avenue"}, {"ln", "lane"})
    if any(len((source_addr & pair) | (target_addr & pair)) > 1 for pair in aliases):
        labels.append("address abbreviation")
    if shared and not labels: labels.append("address reordering/partial overlap")
    if shared and any(addr_df.get((s1["country"], t), 0) > 2500 for t in shared): labels.append("high-frequency address key")
    if not labels: labels.append("other")
    return labels


def run() -> dict:
    start = time.time()
    fd, dbpath = tempfile.mkstemp(prefix="stage4_3_sample_", suffix=".sqlite")
    os.close(fd)
    db = sqlite3.connect(dbpath)
    db.execute("PRAGMA cache_size=-65536")  # bounded 64 MiB SQLite page cache
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA journal_mode=OFF")
    sample, spec = build_sample(db)
    fetch_sample_data(sample)
    assert len(sample) == 300
    assert Counter(x["class"] for x in sample) == {"zero": 100, "one": 100, "multi": 100}
    assert Counter(x["country"] for x in sample) == {"India": 150, "US": 150}
    # Ground-truth/source counts and records needed for missed-positive audit.
    link_source = {}
    for x in sample:
        for tid in x["links"]:
            link_source[tid] = "S2" if tid.startswith("S2-") else "S3"
    queries_p = {(x["country"], x["prefix"]) for x in sample if x["prefix"]}
    queries_a = set().union(*(x["address_keys"] and {(x["country"], t) for t in x["address_keys"]} for x in sample))
    freq_p, freq_a, targets = Counter(), Counter(), {}
    # First target pass: queried-key document frequencies and positive target text.
    for filename in SOURCES:
        source = "S2" if "source2" in filename else "S3"
        for row in rows(DATA / filename):
            pfx, aks = keys_for_target(row)
            if pfx in queries_p: freq_p[pfx] += 1
            for key in aks & queries_a: freq_a[key] += 1
            if row["entity_id"] in link_source:
                targets[row["entity_id"]] = {"source": source, "name": row["business_name"], "business_address": row["business_address"]}
    missing_targets = set(link_source) - set(targets)
    if missing_targets:
        raise RuntimeError(f"Ground-truth target IDs missing: {len(missing_targets)}")
    # Second pass: bounded postings, only selected prefixes and address keys <= max cap.
    pp: dict[tuple[str, str], list[str]] = defaultdict(list)
    ap: dict[tuple[str, str], list[str]] = defaultdict(list)
    maxcap = max(CAPS)
    for filename in SOURCES:
        for row in rows(DATA / filename):
            pfx, aks = keys_for_target(row)
            tid = row["entity_id"]
            if pfx in queries_p:
                pp[pfx].append(tid)
            for key in aks & queries_a:
                if 0 < freq_a[key] <= maxcap:
                    ap[key].append(tid)
    byid = {x["id"]: x for x in sample}
    allmetrics = {}
    all_recovered = {}
    for strat in ("Prefix-5 only", "Address cap 2,500", "Address cap 25,000", "Union cap 2,500", "Union cap 25,000"):
        cap = 2500 if "2,500" in strat else 25000
        counts, recovered, recovered_by_src = {}, set(), Counter()
        count_groups = defaultdict(list)
        zero_values = []
        country_counts = defaultdict(list)
        country_true = Counter()
        country_recovered = Counter()
        for x in sample:
            country_true[x["country"]] += len(x["links"])
            use_prefix = strat.startswith("Prefix") or strat.startswith("Union")
            cands = (set(pp.get((x["country"], x["prefix"]), ()))
                     if use_prefix and x["prefix"] else set())
            if strat != "Prefix-5 only":
                for key in x["address_keys"]:
                    pair = (x["country"], key)
                    if freq_a[pair] <= cap:
                        cands.update(ap.get(pair, ()))
            n = len(cands)
            counts[x["id"]] = n
            count_groups[x["class"]].append(n)
            country_counts[x["country"]].append(n)
            if x["class"] == "zero": zero_values.append(n)
            hit = set(x["links"]) & cands
            recovered.update((x["id"], t) for t in hit)
            for tid in hit:
                src = link_source[tid]
                recovered_by_src[src] += 1
                country_recovered[x["country"]] += 1
        true_s2 = sum(1 for t in link_source.values() if t == "S2")
        true_s3 = sum(1 for t in link_source.values() if t == "S3")
        rec_s2, rec_s3 = recovered_by_src["S2"], recovered_by_src["S3"]
        true = len(link_source)
        misses = [(byid[sid], tid) for sid in byid for tid in byid[sid]["links"] if (sid, tid) not in recovered]
        category = Counter()
        examples = []
        for s1, tid in misses:
            labs = classify_miss(s1, targets[tid], freq_a)
            category.update(labs)
            if len(examples) < 12:
                examples.append({"s1_id": s1["id"], "target_id": tid, "source": targets[tid]["source"],
                                 "s1_name": s1["name"], "target_name": targets[tid]["name"],
                                 "s1_address": s1["address"], "target_address": targets[tid]["business_address"],
                                 "categories": labs})
        bz = Counter()
        for v in zero_values:
            b = "0" if v == 0 else "1-10" if v <= 10 else "11-100" if v <= 100 else "101-1000" if v <= 1000 else "1001-10000" if v <= 10000 else "10001-100000" if v <= 100000 else ">100000"
            bz[b] += 1
        recovered_hash = hashlib.sha256("\n".join(f"{a}\t{b}" for a,b in sorted(recovered)).encode()).hexdigest()
        metrics = {
            "true_links": true, "recovered_links": len(recovered), "missed_links": true-len(recovered),
            "recall": len(recovered)/true if true else 0,
            "s2": {"true": true_s2,"recovered":rec_s2,"missed":true_s2-rec_s2,"recall":rec_s2/true_s2},
            "s3": {"true": true_s3,"recovered":rec_s3,"missed":true_s3-rec_s3,"recall":rec_s3/true_s3},
            "candidate_volume": dist(list(counts.values())),
            "by_match_class": {k: dist(v) for k,v in count_groups.items()},
            "by_country": {k: dist(v) for k,v in country_counts.items()},
            "country_recall": {k: {"true": country_true[k], "recovered": country_recovered[k],
                                   "missed": country_true[k]-country_recovered[k],
                                   "recall": country_recovered[k]/country_true[k]}
                               for k in ("India", "US")},
            "zero_match": {**dist(zero_values), "bins": {k:bz[k] for k in ["0","1-10","11-100","101-1000","1001-10000","10001-100000",">100000"]}},
            "recovered_link_set_sha256": recovered_hash,
            "miss_categories_overlapping": dict(category), "miss_examples": examples,
        }
        if metrics["recovered_links"] != rec_s2 + rec_s3 or true != true_s2+true_s3 or metrics["missed_links"] != true - metrics["recovered_links"]:
            raise AssertionError("recall accounting mismatch")
        for src in ("s2", "s3"):
            if metrics[src]["true"] != metrics[src]["recovered"] + metrics[src]["missed"]:
                raise AssertionError("source accounting mismatch")
        allmetrics[strat] = metrics
        all_recovered[strat] = recovered
    # Sample-level country true links and validation summary.
    true_by_country = Counter(x["country"] for x in sample for _ in x["links"])
    source_true = Counter(link_source.values())
    sample_spec = {
        **spec,
        "sample_n": len(sample),
        "strata": dict(Counter(x["class"] for x in sample)),
        "country_counts": dict(Counter(x["country"] for x in sample)),
        "strata_country_counts": {f"{c}/{k}": sum(1 for x in sample if x["class"]==c and x["country"]==k)
                                  for c in ("zero","one","multi") for k in ("India","US")},
        "true_links": len(link_source), "s2_true_links": source_true["S2"], "s3_true_links": source_true["S3"],
        "country_true_links": dict(true_by_country),
        "validated_target_ids": len(targets), "missing_target_ids": len(missing_targets),
    }
    result = {
        "title": "Stage 4.3 restarted reproducible held-out blocking validation",
        "sample": sample_spec,
        "definitions": {
            "prefix": "country + first 5 characters of Stage 3 conservative normalized business_name; names shorter than 5 have no prefix key",
            "address": "country-specific distinct token keys from the union of conservative and approved abbreviation-normalized addresses; empty address emits no keys; target df counts records in S2+S3 and each record contributes once per key; usable iff 1 <= df <= cap",
            "candidate_count": "unique S2+S3 target records per S1",
            "quantiles": "nearest-rank quantile at ceil(p*n), one-based",
            "address_key_frequency_sample": "all distinct queried country/token keys; frequency counted across both target source files",
        },
        "strategies": allmetrics,
        "union_overlap": {
            "recovered_by_both": len(all_recovered["Union cap 2,500"] & all_recovered["Union cap 25,000"]),
            "recovered_only_by_a": len(all_recovered["Union cap 2,500"] - all_recovered["Union cap 25,000"]),
            "recovered_only_by_b": len(all_recovered["Union cap 25,000"] - all_recovered["Union cap 2,500"]),
            "missed_by_both": len(link_source) - len(all_recovered["Union cap 2,500"] | all_recovered["Union cap 25,000"]),
        },
        "key_frequency": {"queried_address_keys": len(queries_a), "freq_min": min(freq_a.values()) if freq_a else 0,
                          "freq_median": statistics.median(freq_a.values()) if freq_a else 0,
                          "freq_max": max(freq_a.values()) if freq_a else 0,
                          "address_key_df_over_2500_count": sum(v>2500 for v in freq_a.values()),
                          "address_key_df_over_25000_count": sum(v>25000 for v in freq_a.values())},
        "runtime_seconds": round(time.time()-start, 3),
    }
    (ROOT / "experiments" / "stage4_3" / "sample_ids.txt").write_text(
        "\n".join(x["id"] for x in sample), encoding="utf-8")
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    db.close()
    os.unlink(dbpath)
    return result


if __name__ == "__main__":
    run()
