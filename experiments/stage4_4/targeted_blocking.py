#!/usr/bin/env python3
"""Stage 4.4 targeted blocking recovery experiment (training data only).

Run from the project root with:
  python3 experiments/stage4_4/targeted_blocking.py --run-tag run1

The script is intentionally an exploratory experiment, not production code.
It streams the supplied target TSVs twice and retains only queried-key
postings. Outputs are limited to this experiment directory.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import random
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
OUTDIR = ROOT / "experiments" / "stage4_4"
S43_DIR = ROOT / "experiments" / "stage4_3"
SEED = 20261004
SAMPLE_PER_STRATUM = 50
MAX_CAP = 25000
SOURCES = (("train_source2.tsv", "S2"), ("train_source3.tsv", "S3"))
CLASSES = (("zero", 0, 0), ("one", 1, 1), ("multi", 2, 10**9))
COUNTRIES = ("India", "US")
BANDS = ((2500, 5000), (5000, 10000), (10000, 15000), (15000, 25000))

sys.path.insert(0, str(ROOT))
from normalization.normalize import (  # noqa: E402
    normalize_address,
    normalize_address_abbreviations,
    normalize_name,
)


def rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t")


def sha_lines(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


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


def address_keys(value: str | None) -> set[str]:
    a = normalize_address(value) or ""
    b = normalize_address_abbreviations(value) or ""
    return set(a.split()) | set(b.split())


def name_tokens(value: str | None) -> set[str]:
    return set((normalize_name(value) or "").split())


def partition_new(sid: str) -> bool:
    # Stage 4.3's saved sample uses first digest byte <64. Complementary
    # partition guarantees this new sample cannot overlap those 300 IDs.
    return hashlib.sha256(sid.encode("utf-8")).digest()[0] >= 64


def build_new_sample() -> tuple[list[dict], dict]:
    # Temporary SQLite holds only sampling metadata; it is deleted on return.
    with tempfile.NamedTemporaryFile(prefix="stage4_4_sample_", suffix=".sqlite") as tf:
        db = sqlite3.connect(tf.name)
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA synchronous=OFF")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("CREATE TABLE s1(id TEXT PRIMARY KEY,country TEXT,name TEXT,address TEXT)")
        db.execute("CREATE TABLE gt(id TEXT PRIMARY KEY,links TEXT NOT NULL,n INTEGER NOT NULL)")
        db.execute("CREATE INDEX gt_n ON gt(n)")
        for i, r in enumerate(rows(DATA / "train_source1.tsv"), 1):
            db.execute("INSERT INTO s1 VALUES(?,?,?,?)", (r["entity_id"], r["country"], r["business_name"], r["business_address"]))
            if i % 100000 == 0:
                db.commit()
        db.commit()
        for i, r in enumerate(rows(DATA / "train_ground_truth.tsv"), 1):
            raw = r["matched_entity_ids"].strip()
            links = [v.strip() for v in raw.split(",") if v.strip()] if raw else []
            db.execute("INSERT INTO gt VALUES(?,?,?)", (r["source1_entity_id"], json.dumps(links), len(links)))
            if i % 100000 == 0:
                db.commit()
        db.commit()
        rng = random.Random(SEED)
        sample = []
        for cls, low, high in CLASSES:
            for country in COUNTRIES:
                pool = [r[0] for r in db.execute(
                    "SELECT s.id FROM s1 s JOIN gt g ON g.id=s.id "
                    "WHERE s.country=? AND g.n BETWEEN ? AND ? ORDER BY s.id",
                    (country, low, high)) if partition_new(r[0])]
                if len(pool) < SAMPLE_PER_STRATUM:
                    raise RuntimeError(f"Insufficient stratum {cls}/{country}: {len(pool)}")
                for sid in rng.sample(pool, SAMPLE_PER_STRATUM):
                    country0, name, addr = db.execute("SELECT country,name,address FROM s1 WHERE id=?", (sid,)).fetchone()
                    links_json, n = db.execute("SELECT links,n FROM gt WHERE id=?", (sid,)).fetchone()
                    sample.append({"id": sid, "country": country0, "name": name, "address": addr,
                                   "class": cls, "links": json.loads(links_json), "match_count": n})
        class_order = {k: i for i, (k, _, _) in enumerate(CLASSES)}
        country_order = {c: i for i, c in enumerate(COUNTRIES)}
        sample.sort(key=lambda x: (class_order[x["class"]], country_order[x["country"]], x["id"]))
        db.close()

    s43_ids = (S43_DIR / "sample_ids.txt").read_text(encoding="utf-8").splitlines()
    if set(s43_ids) & {x["id"] for x in sample}:
        raise AssertionError("New sample overlaps saved Stage 4.3 IDs")
    for x in sample:
        x["norm_name"] = normalize_name(x["name"]) or ""
        x["prefix4"] = x["norm_name"][:4] if len(x["norm_name"]) >= 4 else ""
        x["prefix5"] = x["norm_name"][:5] if len(x["norm_name"]) >= 5 else ""
        x["prefix6"] = x["norm_name"][:6] if len(x["norm_name"]) >= 6 else ""
        x["addr_keys"] = address_keys(x["address"])
        x["name_keys"] = name_tokens(x["name"])
    ids = [x["id"] for x in sample]
    spec = {
        "seed": SEED,
        "algorithm": "stdlib random.Random.sample without replacement; one RNG across strata",
        "source_order": "population sorted lexicographically by S1 ID within exact match-class × country stratum",
        "filtering": "join supplied train_source1.tsv and train_ground_truth.tsv by ID; exact country India/US; match classes 0, 1, >=2; retain SHA-256(UTF-8 S1 ID) first byte >=64",
        "stratum_order": "zero, one, multi; within each: India then US",
        "quota": "50 S1 IDs per match-class × country stratum",
        "final_id_order": "match class zero/one/multi, then India/US, then S1 ID lexicographically",
        "sample_count": len(ids), "sample_sha256": sha_lines(ids),
        "hash_input": "UTF-8 final ordered S1 IDs joined by LF, no trailing LF",
        "disjointness": "The Stage 4.3 saved sample uses the complementary SHA-256 first-byte partition (<64); direct ID-set overlap also checked.",
        "strata": dict(Counter(x["class"] for x in sample)),
        "country_counts": dict(Counter(x["country"] for x in sample)),
        "strata_country_counts": {f"{c}/{k}": sum(x["class"] == c and x["country"] == k for x in sample)
                                  for c, _, _ in CLASSES for k in COUNTRIES},
        "true_links": sum(len(x["links"]) for x in sample),
        "s2_true_links": sum(t.startswith("S2-") for x in sample for t in x["links"]),
        "s3_true_links": sum(t.startswith("S3-") for x in sample for t in x["links"]),
    }
    return sample, spec


def load_stage43() -> tuple[list[dict], dict]:
    p = S43_DIR / "latest_results.json"
    saved = json.loads(p.read_text(encoding="utf-8"))
    ids = (S43_DIR / "sample_ids.txt").read_text(encoding="utf-8").splitlines()
    if len(ids) != 300 or sha_lines(ids) != saved["sample"]["sample_hash"]:
        raise AssertionError("Saved Stage 4.3 sample IDs/hash disagree")
    need = set(ids)
    gt, meta = {}, {}
    for r in rows(DATA / "train_source1.tsv"):
        if r["entity_id"] in need:
            meta[r["entity_id"]] = {"id": r["entity_id"], "country": r["country"], "name": r["business_name"], "address": r["business_address"]}
    for r in rows(DATA / "train_ground_truth.tsv"):
        sid = r["source1_entity_id"]
        if sid in need:
            raw = r["matched_entity_ids"].strip()
            links = [v.strip() for v in raw.split(",") if v.strip()] if raw else []
            gt[sid] = links
    if set(meta) != need or set(gt) != need:
        raise AssertionError("Saved Stage 4.3 sample IDs not fully found in supplied training files")
    sample = []
    for sid in ids:
        x = meta[sid]
        x["links"] = gt[sid]
        x["match_count"] = len(x["links"])
        x["class"] = "zero" if not x["links"] else "one" if len(x["links"]) == 1 else "multi"
        x["norm_name"] = normalize_name(x["name"]) or ""
        x["prefix4"] = x["norm_name"][:4] if len(x["norm_name"]) >= 4 else ""
        x["prefix5"] = x["norm_name"][:5] if len(x["norm_name"]) >= 5 else ""
        x["prefix6"] = x["norm_name"][:6] if len(x["norm_name"]) >= 6 else ""
        x["addr_keys"] = address_keys(x["address"])
        x["name_keys"] = name_tokens(x["name"])
        sample.append(x)
    if sum(len(x["links"]) for x in sample) != 487:
        raise AssertionError("Saved Stage 4.3 sample no longer has 487 links")
    return sample, saved


def target_keys(r: dict) -> tuple[dict[int, tuple[str, str]], set[str], set[str]]:
    c = r["country"]
    n = normalize_name(r["business_name"]) or ""
    pfx = {k: (c, n[:k]) for k in (4, 5, 6) if len(n) >= k}
    return pfx, address_keys(r["business_address"]), name_tokens(r["business_name"])


def candidate_rule_results(sample: list[dict], ppm: dict, apm: dict, nam: dict,
                           dfp: dict, dfa: dict, dfn: dict, target_info: dict,
                           all_links: dict[str, str], rules: tuple[str, ...]) -> tuple[dict, dict]:
    recovered_sets = {}
    metrics = {}
    rule_labels = {
        "A": "prefix5_or_address_le2500",
        "B": "prefix5_or_address_le25000",
        "PREFIX4": "prefix4_only",
        "PREFIX6": "prefix6_only",
        "NAME2500": "name_token_df_le2500_only",
        "NAME25000": "name_token_df_le25000_only",
        "A_NAME2500": "A_plus_name_token_df_le2500",
        "A_NAME25000": "A_plus_name_token_df_le25000",
        "A_HF_PREFIX": "A_plus_hf_address_and_prefix5",
        "A_HF_ANYADDR": "A_plus_hf_address_and_second_address_token",
        "A_HF_RARE": "A_plus_hf_address_and_rare_address_token",
        "A_HF_NAME": "A_plus_hf_address_and_name_token",
        "A_TWOADDR": "A_plus_two_shared_address_tokens",
    }
    if any(r.startswith("A_HF_BAND") for r in rules):
        for hi, (lo, top) in enumerate(BANDS):
            rules0 = f"A_HF_BAND{hi+1}_ANYADDR"
            rule_labels[rules0] = f"A_plus_hf_address_band_{lo+1}_{top}_and_second_address_token"
            rules += (rules0,)
    # Per S1 candidates and recovered links, with the extra block logic
    # constrained to queried keys and capped postings.
    per_rule_counts = {r: [] for r in rules}
    per_rule_zero = {r: [] for r in rules}
    per_rule_hit = {r: set() for r in rules}
    per_rule_extra = {r: [] for r in rules}
    per_rule_extra_zero = {r: [] for r in rules}
    # Only per-strategy positive hits are retained; avoid keeping candidate
    # sets for every sample row in memory.
    for x in sample:
        c = x["country"]
        p5 = set(ppm.get((c, 5, x["prefix5"]), ())) if x["prefix5"] else set()
        p4 = set(ppm.get((c, 4, x["prefix4"]), ())) if x["prefix4"] else set()
        p6 = set(ppm.get((c, 6, x["prefix6"]), ())) if x["prefix6"] else set()
        addr_union = set()
        hf_counts = Counter()
        rare_counts = Counter()
        band_counts = [Counter() for _ in BANDS]
        address_token_counts = Counter()
        for tok in x["addr_keys"]:
            key = (c, tok)
            df = dfa.get(key, 0)
            if 0 < df <= MAX_CAP:
                ids = apm.get(key, ())
                if ids:
                    address_token_counts.update(ids)
                    addr_union.update(ids)
                    if df > 2500:
                        hf_counts.update(ids)
                        for i, (a, b) in enumerate(BANDS):
                            if a < df <= b:
                                band_counts[i].update(ids)
                    else:
                        rare_counts.update(ids)
        name_union = set()
        for tok in x["name_keys"]:
            key = (c, tok)
            if 0 < dfn.get(key, 0) <= MAX_CAP:
                name_union.update(nam.get(key, ()))
        a_base = set(p5)
        for tok in x["addr_keys"]:
            if 0 < dfa.get((c, tok), 0) <= 2500:
                a_base.update(apm.get((c, tok), ()))
        b_base = set(p5) | addr_union
        hf_targets = set(hf_counts)
        rare_targets = set(rare_counts)
        hf_prefix = hf_targets & p5
        hf_anyaddr = {tid for tid, n in address_token_counts.items() if n >= 2 and tid in hf_targets}
        hf_rare = hf_targets & rare_targets
        hf_name = hf_targets & name_union
        two_addr = {tid for tid, n in address_token_counts.items() if n >= 2}
        candidates = {
            "A": a_base,
            "B": b_base,
            "PREFIX4": p4,
            "PREFIX6": p6,
            "NAME2500": set().union(*(nam.get((c, tok), ()) for tok in x["name_keys"] if 0 < dfn.get((c, tok), 0) <= 2500)),
            "NAME25000": name_union,
            "A_NAME2500": set(a_base) | set().union(*(nam.get((c, tok), ()) for tok in x["name_keys"] if 0 < dfn.get((c, tok), 0) <= 2500)),
            "A_NAME25000": a_base | name_union,
            "A_HF_PREFIX": a_base | hf_prefix,
            "A_HF_ANYADDR": a_base | hf_anyaddr,
            "A_HF_RARE": a_base | hf_rare,
            "A_HF_NAME": a_base | hf_name,
            "A_TWOADDR": a_base | two_addr,
        }
        for i in range(len(BANDS)):
            cand = set()
            for tok in x["addr_keys"]:
                key = (c, tok)
                df = dfa.get(key, 0)
                if BANDS[i][0] < df <= BANDS[i][1]:
                    ids = apm.get(key, ())
                    # Another distinct eligible address token must also occur.
                    cand.update(tid for tid in ids if address_token_counts.get(tid, 0) >= 2)
            candidates[f"A_HF_BAND{i+1}_ANYADDR"] = a_base | cand
        for r in rules:
            cs = candidates[r]
            per_rule_counts[r].append(len(cs))
            if x["class"] == "zero":
                per_rule_zero[r].append(len(cs))
            hits = {(x["id"], tid) for tid in x["links"] if tid in cs}
            per_rule_hit[r].update(hits)
            extra = cs - a_base
            per_rule_extra[r].append(len(extra))
            if x["class"] == "zero":
                per_rule_extra_zero[r].append(len(extra))

    nlinks = len(all_links)
    source_true = Counter(all_links.values())
    for r in rules:
        hits = per_rule_hit[r]
        source_rec = Counter(all_links[t] for _, t in hits)
        zero = per_rule_zero[r]
        bins = Counter("0" if v == 0 else "1-10" if v <= 10 else "11-100" if v <= 100 else "101-1000" if v <= 1000 else "1001-10000" if v <= 10000 else "10001-100000" if v <= 100000 else ">100000" for v in zero)
        metrics[r] = {
            "label": rule_labels[r], "true_links": nlinks,
            "recovered_links": len(hits), "missed_links": nlinks-len(hits),
            "recall": len(hits)/nlinks,
            "s2": {"true": source_true["S2"], "recovered": source_rec["S2"], "missed": source_true["S2"]-source_rec["S2"], "recall": source_rec["S2"]/source_true["S2"]},
            "s3": {"true": source_true["S3"], "recovered": source_rec["S3"], "missed": source_true["S3"]-source_rec["S3"], "recall": source_rec["S3"]/source_true["S3"]},
            "candidate_volume": dist(per_rule_counts[r]),
            "zero_match": {**dist(zero), "bins": {k: bins[k] for k in ("0", "1-10", "11-100", "101-1000", "1001-10000", "10001-100000", ">100000")}},
            "recovered_link_set_sha256": hashlib.sha256("\n".join(f"{a}\t{b}" for a,b in sorted(hits)).encode()).hexdigest(),
            "marginal_vs_A": {
                "additional_links": len(hits - per_rule_hit["A"]),
                "added_candidate_pairs": sum(per_rule_counts[r]) - sum(per_rule_counts["A"]),
                "added_candidates_per_additional_link": ((sum(per_rule_counts[r])-sum(per_rule_counts["A"])) / len(hits-per_rule_hit["A"])) if hits-per_rule_hit["A"] else None,
                "mean_additional_candidates_per_s1": statistics.fmean(per_rule_extra[r]) if per_rule_extra[r] else 0,
                "p95_additional_candidates_per_s1": qtile(per_rule_extra[r], .95),
                "zero_match_additional_candidate_pairs": sum(per_rule_extra_zero[r]),
                "zero_match_mean_additional_candidates": statistics.fmean(per_rule_extra_zero[r]) if per_rule_extra_zero[r] else 0,
            },
        }
        if metrics[r]["recovered_links"] != metrics[r]["s2"]["recovered"] + metrics[r]["s3"]["recovered"]:
            raise AssertionError(f"S2/S3 accounting failed for {r}")
    return metrics, per_rule_hit


def compare_repro(a: dict, b: dict) -> bool:
    def strip_runtime(obj):
        if isinstance(obj, dict):
            return {k: strip_runtime(v) for k, v in obj.items() if k not in {"runtime_seconds", "elapsed_seconds"}}
        if isinstance(obj, list):
            return [strip_runtime(x) for x in obj]
        return obj
    return strip_runtime(a) == strip_runtime(b)


def run(tag: str) -> dict:
    t0 = time.time()
    s43, saved43 = load_stage43()
    new, newspec = build_new_sample()
    if not all(hashlib.sha256(x["id"].encode()).digest()[0] < 64 for x in s43):
        raise AssertionError("The saved Stage 4.3 sample is outside its documented hash partition")
    print(f"sample-ready tag={tag} new_hash={newspec['sample_sha256']} links={newspec['true_links']}", flush=True)
    if set(x["id"] for x in s43) & set(x["id"] for x in new):
        raise AssertionError("Stage 4.3 / 4.4 sample overlap")
    all_samples = s43 + new
    link_source = {}
    source_info = {x["id"]: x for x in all_samples}
    all_links = {}
    for x in all_samples:
        for tid in x["links"]:
            src = "S2" if tid.startswith("S2-") else "S3" if tid.startswith("S3-") else "UNKNOWN"
            if src == "UNKNOWN":
                raise AssertionError(f"Unknown target ID prefix: {tid}")
            if tid in all_links and all_links[tid] != src:
                raise AssertionError(f"Target ID source collision {tid}")
            all_links[tid] = src
            link_source[tid] = src
    queries_p = {(x["country"], k, x[f"prefix{k}"]) for x in all_samples for k in (4, 5, 6) if x[f"prefix{k}"]}
    queries_a = {(x["country"], tok) for x in all_samples for tok in x["addr_keys"]}
    queries_n = {(x["country"], tok) for x in all_samples for tok in x["name_keys"]}
    dfp, dfa, dfn = Counter(), Counter(), Counter()
    positive_info = {}
    # First streamed target pass: exact sample-query df and positive data.
    for filename, src in SOURCES:
        for r in rows(DATA / filename):
            c = r["country"]
            pfx, aks, ntoks = target_keys(r)
            for k, pair in pfx.items():
                if (pair[0], k, pair[1]) in queries_p:
                    dfp[(pair[0], k, pair[1])] += 1
            for tok in aks:
                key = (c, tok)
                if key in queries_a:
                    dfa[key] += 1
            for tok in ntoks:
                key = (c, tok)
                if key in queries_n:
                    dfn[key] += 1
            if r["entity_id"] in all_links:
                positive_info[r["entity_id"]] = {
                    "source": src, "country": c, "name": r["business_name"],
                    "address": r["business_address"], "norm_name": normalize_name(r["business_name"]) or "",
                    "addr_keys": aks, "name_keys": ntoks,
                }
    if set(positive_info) != set(all_links):
        raise RuntimeError(f"Missing positive targets: {len(set(all_links)-set(positive_info))}")
    print("frequency-scan-complete", flush=True)
    ppm: dict[tuple[str, int, str], list[str]] = defaultdict(list)
    apm: dict[tuple[str, str], list[str]] = defaultdict(list)
    nam: dict[tuple[str, str], list[str]] = defaultdict(list)
    # Second pass: only keys actually queried by either bounded sample.
    for filename, _ in SOURCES:
        for r in rows(DATA / filename):
            tid, c = r["entity_id"], r["country"]
            pfx, aks, ntoks = target_keys(r)
            for k, pair in pfx.items():
                key = (pair[0], k, pair[1])
                if key in queries_p:
                    ppm[key].append(tid)
            for tok in aks:
                key = (c, tok)
                if key in queries_a and 0 < dfa[key] <= MAX_CAP:
                    apm[key].append(tid)
            for tok in ntoks:
                key = (c, tok)
                if key in queries_n and 0 < dfn[key] <= MAX_CAP:
                    nam[key].append(tid)
    print("posting-scan-complete", flush=True)

    # Split sample query-key views while sharing the target postings.
    s43_ids = {x["id"] for x in s43}
    new_ids = {x["id"] for x in new}
    pos_s43 = {tid: src for tid, src in all_links.items() if tid in {t for x in s43 for t in x["links"]}}
    pos_new = {tid: src for tid, src in all_links.items() if tid in {t for x in new for t in x["links"]}}
    explor_rules = ("A", "B", "PREFIX4", "PREFIX6", "NAME2500", "NAME25000", "A_NAME2500", "A_NAME25000", "A_HF_PREFIX", "A_HF_ANYADDR", "A_HF_RARE", "A_HF_NAME", "A_TWOADDR", *(f"A_HF_BAND{i}_ANYADDR" for i in range(1, 5)))
    metrics43, hit43 = candidate_rule_results(s43, ppm, apm, nam, dfp, dfa, dfn, positive_info, pos_s43, explor_rules)
    gate_values = (metrics43["A"]["recovered_links"], metrics43["A"]["s2"]["recovered"], metrics43["A"]["s3"]["recovered"],
                   metrics43["B"]["recovered_links"], metrics43["B"]["s2"]["recovered"], metrics43["B"]["s3"]["recovered"])
    if gate_values != (460, 234, 226, 484, 251, 233):
        raise RuntimeError(f"Stage 4.3 reproduction gate failed: {gate_values}")
    volume_gate = {
        "A": (1932066, 6440.22, 20794, 30685),
        "B": (7390062, 24633.54, 58191, 73224),
    }
    for rule, expected in volume_gate.items():
        observed = metrics43[rule]["candidate_volume"]
        got = (observed["total"], observed["mean"], observed["p95"], observed["p99"])
        if got != expected:
            raise RuntimeError(f"Stage 4.3 candidate-volume gate failed for {rule}: {got} != {expected}")
    ra, rb = hit43["A"], hit43["B"]
    if (len(ra & rb), len(ra-rb), len(rb-ra), 487-len(ra|rb)) != (460, 0, 24, 3):
        raise RuntimeError("Stage 4.3 overlap gate failed")
    print("stage43-gate-passed A=460/487 B=484/487 B_only=24 missed_both=3", flush=True)
    # Select only rules supported by Stage 4.3 diagnostics: HF-address plus a
    # name token (compact marginal volume), two-address agreement, and the
    # lower name-token cap that recovered the three empty-address misses.
    val_rules = ("A", "B", "A_HF_NAME", "A_TWOADDR", "A_NAME2500")
    metrics_new, hit_new = candidate_rule_results(new, ppm, apm, nam, dfp, dfa, dfn, positive_info, pos_new, val_rules)
    # Positive-link diagnostics on old sample: full record-level audit for
    # B-only links and all persistent B misses.
    b_only = sorted(rb-ra)
    missed_b = sorted({(x["id"], tid) for x in s43 for tid in x["links"]} - rb)
    by_s43 = {x["id"]: x for x in s43}
    def link_detail(sid: str, tid: str, status: str) -> dict:
        x, t = by_s43[sid], positive_info[tid]
        common_addr = sorted(x["addr_keys"] & t["addr_keys"])
        common_name = sorted(x["name_keys"] & t["name_keys"])
        df = {tok: dfa.get((x["country"], tok), 0) for tok in common_addr}
        tokfreq = {tok: dfn.get((x["country"], tok), 0) for tok in common_name}
        bins = {"<=2500": [], "2501-5000": [], "5001-10000": [], "10001-15000": [], "15001-25000": [], ">25000_or_absent": []}
        for tok, v in df.items():
            key = "<=2500" if 0 < v <= 2500 else "2501-5000" if v <= 5000 else "5001-10000" if v <= 10000 else "10001-15000" if v <= 15000 else "15001-25000" if v <= 25000 else ">25000_or_absent"
            bins[key].append(tok)
        scripts = lambda s: sorted({unicodedata.name(ch, "").split(" ")[0] for ch in s if ch.isalpha()})
        ratio = difflib.SequenceMatcher(None, x["norm_name"], t["norm_name"]).ratio()
        prefix5 = bool(x["prefix5"] and t["norm_name"][:5] == x["prefix5"])
        prefix_match = {str(k): bool(x[f"prefix{k}"] and t["country"] == x["country"] and t["norm_name"][:k] == x[f"prefix{k}"])
                        for k in (4, 5, 6)}
        usable_shared = [tok for tok, v in df.items() if 0 < v <= MAX_CAP]
        return {
            "status": status, "s1_id": sid, "target_source": t["source"], "target_id": tid,
            "country_s1": x["country"], "country_target": t["country"],
            "s1_name": x["name"], "target_name": t["name"],
            "s1_normalized_name": x["norm_name"], "target_normalized_name": t["norm_name"],
            "prefix5": x["prefix5"], "prefix5_matches": prefix5,
            "prefix4_5_6_matches_same_country": prefix_match,
            "s1_address": x["address"], "target_address": t["address"],
            "target_address_missing": not bool(t["address"].strip()),
            "s1_address_tokens": sorted(x["addr_keys"]), "target_address_tokens": sorted(t["addr_keys"]),
            "shared_address_tokens": common_addr, "shared_address_token_df": df,
            "shared_address_frequency_bands": bins,
            "shared_name_tokens": common_name, "shared_name_token_df": tokfreq,
            "multiple_shared_address_tokens": len(common_addr) >= 2,
            "shared_eligible_address_token_count": len(usable_shared),
            "two_shared_eligible_address_tokens": len(usable_shared) >= 2,
            "any_shared_address_df_le2500": any(0 < v <= 2500 for v in df.values()),
            "name_script_sets": {"s1": scripts(x["norm_name"]), "target": scripts(t["norm_name"])},
            "name_token_sets_equal": set(x["norm_name"].split()) == set(t["norm_name"].split()),
            "name_token_jaccard": (len(set(x["norm_name"].split()) & set(t["norm_name"].split())) /
                                   len(set(x["norm_name"].split()) | set(t["norm_name"].split())))
                                  if set(x["norm_name"].split()) | set(t["norm_name"].split()) else 0,
            "name_sequence_ratio_descriptive_only": ratio,
        }
    b_only_details = [link_detail(s, t, "B_ONLY") for s, t in b_only]
    persistent_details = [link_detail(s, t, "MISSED_BY_A_AND_B") for s, t in missed_b]
    targeted_rules = ("A_HF_PREFIX", "A_HF_ANYADDR", "A_HF_RARE", "A_HF_NAME", "A_TWOADDR",
                      "A_NAME2500", "A_NAME25000", *(f"A_HF_BAND{i}_ANYADDR" for i in range(1, 5)))
    for d in b_only_details + persistent_details:
        pair = (d["s1_id"], d["target_id"])
        d["targeted_rule_recovery"] = {r: pair in hit43[r] for r in targeted_rules}
    # Validation miss sample with positive records for interpretation.
    miss_ex = {}
    for r in val_rules:
        hits = hit_new[r]
        misspairs = sorted({(x["id"], tid) for x in new for tid in x["links"]} - hits)
        examples = []
        categories = Counter()
        for sid, tid in misspairs:
            x = next(z for z in new if z["id"] == sid)
            t = positive_info[tid]
            ta, tb = set(x["norm_name"].split()), set(t["norm_name"].split())
            scripts = lambda s: {unicodedata.name(ch, "").split(" ")[0] for ch in s if ch.isalpha()}
            cats = []
            if not t["address"].strip(): cats.append("missing address")
            if scripts(x["norm_name"]) != scripts(t["norm_name"]): cats.append("multilingual/script")
            if ta == tb and x["norm_name"] != t["norm_name"]: cats.append("name order/punctuation")
            ratio = difflib.SequenceMatcher(None, x["norm_name"], t["norm_name"]).ratio()
            if ratio >= .72 and x["norm_name"] != t["norm_name"]: cats.append("name typo/abbreviation")
            if x["prefix5"] and t["norm_name"][:5] == x["prefix5"]: cats.append("prefix5 hit; address rule")
            shared = sorted(x["addr_keys"] & t["addr_keys"])
            if shared and any(dfa.get((x["country"], tok), 0) > 2500 for tok in shared): cats.append("high-frequency address")
            if shared and not cats: cats.append("address variation")
            if not cats: cats.append("other")
            categories.update(cats)
            if len(examples) < 15:
                examples.append({"s1_id": sid, "target_id": tid, "source": t["source"], "country": x["country"],
                                 "s1_name": x["name"], "target_name": t["name"], "s1_address": x["address"],
                                 "target_address": t["address"], "categories": cats})
        miss_ex[r] = {"missed_links": len(misspairs), "category_counts_overlapping": dict(categories), "examples": examples}

    # For diagnostics: frequency bands of each key responsible for B-only.
    hf_cause = Counter()
    targeted_recovery_bonly = {}
    for r in ("A_HF_PREFIX", "A_HF_ANYADDR", "A_HF_RARE", "A_HF_NAME", "A_TWOADDR", *(f"A_HF_BAND{i}_ANYADDR" for i in range(1, 5))):
        rec = hit43[r]
        targeted_recovery_bonly[r] = len((rec-ra) & (rb-ra))
    for d in b_only_details:
        for band, toks in d["shared_address_frequency_bands"].items():
            if toks and band in ("2501-5000", "5001-10000", "10001-15000", "15001-25000"):
                hf_cause[band] += 1
    result = {
        "title": "Stage 4.4 targeted blocking recovery experiment",
        "run_tag": tag,
        "stage43_baseline_artifact": {
            "sample_sha256": saved43["sample"]["sample_hash"],
            "sample_count": len(s43), "true_links": sum(len(x["links"]) for x in s43),
            "saved_A_recovered": saved43["strategies"]["Union cap 2,500"]["recovered_links"],
            "saved_B_recovered": saved43["strategies"]["Union cap 25,000"]["recovered_links"],
            "recomputed_A_recovered": metrics43["A"]["recovered_links"],
            "recomputed_B_recovered": metrics43["B"]["recovered_links"],
            "recomputed_overlap": {"both": len(ra&rb), "A_only": len(ra-rb), "B_only": len(rb-ra), "missed_both": 487-len(ra|rb)},
            "gate_passed": True,
        },
        "new_sample": newspec,
        "definitions": {
            "prefix": "country + first 4/5/6 chars of Stage 3 conservative-normalized name; short strings do not emit that prefix",
            "address": "country + each distinct key from union of conservative/approved abbreviation address tokens; df is target-record count S2+S3; use 1..cap",
            "name": "country + distinct conservative normalized name token; df is target-record count S2+S3; use 1..cap",
            "high_frequency_band": "2500 < address-token df <= 25000; target record must share one key in the rule band",
            "candidate_count": "unique target entity IDs across S2 and S3 for each S1",
            "quantile": "nearest rank ceil(p*n)",
        },
        "stage43_exploration": {
            "metrics": metrics43,
            "b_only_count": len(b_only_details), "b_only_frequency_band_counts": dict(hf_cause),
            "b_only_rule_recovery_counts": targeted_recovery_bonly,
            "b_only_links": b_only_details,
            "persistent_miss_count": len(persistent_details), "persistent_misses": persistent_details,
        },
        "new_validation": {
            "selected_after_stage43_exploration": list(val_rules),
            "strategies": metrics_new,
            "miss_analysis": miss_ex,
        },
        "runtime_seconds": round(time.time()-t0, 3),
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "sample_ids.txt").write_text("\n".join(x["id"] for x in new)+"\n", encoding="utf-8")
    (OUTDIR / f"{tag}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True, choices=("run1", "run2"))
    args = parser.parse_args()
    output = run(args.run_tag)
    print(json.dumps(output, ensure_ascii=False, indent=2))
