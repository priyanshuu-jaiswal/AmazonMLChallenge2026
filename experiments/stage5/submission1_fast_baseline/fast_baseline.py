#!/usr/bin/env python3
"""Storage-conscious deterministic direct-match baseline for Submission #1.

Builds compact SQLite postings over test S2/S3, then streams test S1 rows.
It does not implement or replace the frozen Stage 4 blocker.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TEST = ROOT / "student_resource/dataset/test"
HERE = Path(__file__).resolve().parent
DB = HERE / "target_index.sqlite"
OUT = ROOT / "output"
sys.path.insert(0, str(ROOT))
from normalization.normalize import normalize_name, normalize_address, normalize_address_abbreviations

TABLES = ("by_name", "by_address", "by_pair", "by_abbr_address")


def key(value: str | None) -> str:
    # JSON tuple encoding avoids delimiter collisions in arbitrary supplied text.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def index_targets(db: sqlite3.Connection) -> dict:
    start = time.time()
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-262144")  # cap SQLite page cache near 256 MiB
    for table in TABLES:
        db.execute(f"CREATE TABLE IF NOT EXISTS {table} (country TEXT NOT NULL, k TEXT NOT NULL, tid TEXT NOT NULL, PRIMARY KEY(country,k,tid)) WITHOUT ROWID")
    db.commit()
    counts = {table: 0 for table in TABLES}
    insert_sql = {table: f"INSERT OR IGNORE INTO {table}(country,k,tid) VALUES(?,?,?)" for table in TABLES}
    for filename in ("test_source2.tsv", "test_source3.tsv"):
        with open(TEST / filename, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            batch = {table: [] for table in TABLES}
            for row in reader:
                country, tid = row["country"], row["entity_id"]
                name = normalize_name(row["business_name"])
                address = normalize_address(row["business_address"])
                abbr = normalize_address_abbreviations(row["business_address"])
                vals = {
                    "by_name": (country, key(name), tid),
                    "by_address": (country, key(address), tid),
                    "by_pair": (country, key([name, address]), tid),
                    "by_abbr_address": (country, key([name, abbr]), tid),
                }
                for table, value in vals.items():
                    # Empty/missing names or addresses are not valid exact evidence.
                    if name and (table == "by_name" or (address if table in ("by_address", "by_pair") else abbr)):
                        batch[table].append(value)
                if len(batch["by_name"]) >= 5000:
                    for table in TABLES:
                        db.executemany(insert_sql[table], batch[table])
                        counts[table] += len(batch[table])
                        batch[table].clear()
            for table in TABLES:
                db.executemany(insert_sql[table], batch[table])
                counts[table] += len(batch[table])
    db.commit()
    return {"index_build_seconds": round(time.time() - start, 3), "indexed_posting_rows": counts}


def fetch(db, table, country, k):
    return [r[0] for r in db.execute(f"SELECT tid FROM {table} WHERE country=? AND k=?", (country, k))]


def direct_matches(db, row):
    country = row["country"]
    name, address = normalize_name(row["business_name"]), normalize_address(row["business_address"])
    abbr = normalize_address_abbreviations(row["business_address"])
    if not name:
        return []
    # Rules 1/2: exact normalized name plus exact conservative/abbreviation address.
    accepted = set()
    if address:
        accepted.update(fetch(db, "by_pair", country, key([name, address])))
        accepted.update(fetch(db, "by_abbr_address", country, key([name, abbr])))
    # Rule 4: accept only a globally unique (within this country and both target files)
    # exact normalized name; ambiguity is not resolved arbitrarily.
    if not accepted:
        ids = fetch(db, "by_name", country, key(name))
        if len(ids) == 1:
            accepted.update(ids)
    return sorted(accepted)


def stream_source1(limit=None, outputs=None, db=None):
    start = time.time()
    path = TEST / "test_source1.tsv"
    if outputs is None:
        outputs = (None, None)
    mfile, cfile = outputs
    mw = csv.writer(mfile, delimiter="\t", lineterminator="\n") if mfile else None
    cw = csv.writer(cfile, delimiter="\t", lineterminator="\n") if cfile else None
    if mw: mw.writerow(["source1_entity_id", "matched_entity_ids"])
    if cw: cw.writerow(["source1_entity_id", "candidate_entity_ids"])
    rows = predicted = zero = one = many = 0
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ids = direct_matches(db, row)
            rows += 1
            predicted += len(ids)
            if len(ids) == 0: zero += 1
            elif len(ids) == 1: one += 1
            else: many += 1
            if mw: mw.writerow([row["entity_id"], ",".join(ids)])
            if cw: cw.writerow([row["entity_id"], ",".join(ids)])
            if limit is not None and rows >= limit:
                break
    elapsed = time.time() - start
    return {"rows_processed": rows, "predicted_match_ids": predicted,
            "distribution_s1_rows_zero_one_many": {"zero": zero, "one": one, "many": many},
            "rows_per_second": round(rows / elapsed, 2) if elapsed else None,
            "elapsed_seconds": round(elapsed, 3)}


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--benchmark", type=int, metavar="ROWS")
    mode.add_argument("--full", action="store_true")
    ap.add_argument("--rebuild-index", action="store_true")
    args = ap.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    if args.rebuild_index and DB.exists():
        DB.unlink()
    existed = DB.exists()
    db = sqlite3.connect(DB)
    build = {"index_reused": existed}
    if not existed:
        build.update(index_targets(db))
    if args.benchmark:
        result = stream_source1(args.benchmark, db=db)
        report = {**build, **result,
                  "db_bytes": DB.stat().st_size,
                  "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)}
        (HERE / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    OUT.mkdir(exist_ok=True)
    with open(OUT / "matching_results.tsv", "w", encoding="utf-8", newline="") as mf, open(OUT / "candidate_pairs.tsv", "w", encoding="utf-8", newline="") as cf:
        result = stream_source1(outputs=(mf, cf), db=db)
    report = {**build, **result,
              "matching_results_bytes": (OUT / "matching_results.tsv").stat().st_size,
              "candidate_pairs_bytes": (OUT / "candidate_pairs.tsv").stat().st_size,
              "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)}
    (HERE / "result_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    db.close()


if __name__ == "__main__":
    main()
