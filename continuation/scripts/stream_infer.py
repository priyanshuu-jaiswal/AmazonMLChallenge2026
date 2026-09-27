#!/usr/bin/env python3
"""Stream frozen Stage 4 candidates through the Stage 5 scorer for test S1s.

The candidate TSV is written in the required plain-text format and contains
every unique candidate passed to the scorer. This script is intentionally not
run automatically: its storage preflight should be reviewed first.
"""
import argparse, csv, shutil, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
TEST=ROOT/'student_resource/dataset/test'
OUT=ROOT/'output'
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'experiments/stage5'))
from candidate_generator import read_rows, build_index, candidates
from features import pair_features
from scorer import score

THRESHOLD=12.0
ESTIMATED_PAIRS_PER_S1=7311.448666666666  # Stage 4.7 sample mean; estimate only.
BYTES_PER_CANDIDATE=13  # 12-byte entity ID plus comma, conservative TSV-list estimate.
RESERVE_BYTES=1<<30

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--preflight-only',action='store_true',help='estimate output size and check free space, without loading data')
    ap.add_argument('--allow-large-output',action='store_true',help='bypass estimated-size guard; does not create extra disk capacity')
    args=ap.parse_args()
    s1_path=TEST/'test_source1.tsv'
    with open(s1_path,encoding='utf-8',newline='') as f:
        n_s1=sum(1 for _ in f)-1
    required=int(n_s1*ESTIMATED_PAIRS_PER_S1*BYTES_PER_CANDIDATE)
    free=shutil.disk_usage(ROOT).free
    print(f'test_s1_rows={n_s1}')
    print(f'estimated_candidate_pairs={int(n_s1*ESTIMATED_PAIRS_PER_S1)} (training-sample extrapolation, not measured)')
    print(f'estimated_candidate_tsv_bytes={required}')
    print(f'free_bytes={free}')
    if args.preflight_only: return
    if not args.allow_large_output and required+RESERVE_BYTES>free:
        raise SystemExit('Storage preflight refused full inference: estimated plain candidate_pairs.tsv plus 1 GiB reserve exceeds free space.')

    s1_rows=list(read_rows(s1_path))
    targets=[TEST/'test_source2.tsv',TEST/'test_source3.tsv']
    idx=build_index(s1_rows,targets)
    OUT.mkdir(parents=True,exist_ok=True)
    match_path=OUT/'matching_results.tsv'; cand_path=OUT/'candidate_pairs.tsv'
    total=predicted=0; started=time.time()
    with open(match_path,'w',encoding='utf-8',newline='') as mf, open(cand_path,'w',encoding='utf-8',newline='') as cf:
        mw=csv.writer(mf,delimiter='\t',lineterminator='\n'); cw=csv.writer(cf,delimiter='\t',lineterminator='\n')
        mw.writerow(['source1_entity_id','matched_entity_ids'])
        cw.writerow(['source1_entity_id','candidate_entity_ids'])
        for s1 in s1_rows:
            ids=sorted(candidates(s1,idx))  # candidate generator already deduplicates the union.
            total+=len(ids)
            accepted=[]
            for tid in ids:
                ft=pair_features(s1,idx['target_info'][tid])
                if score(ft)>=THRESHOLD: accepted.append(tid)
            predicted+=len(accepted)
            cw.writerow([s1['entity_id'],','.join(ids)])
            mw.writerow([s1['entity_id'],','.join(accepted)])
            if total and total%10_000_000 < len(ids):
                mf.flush();cf.flush()
    print(f'test_candidate_pairs={total}')
    print(f'test_predicted_matches={predicted}')
    print(f'matching_results_bytes={match_path.stat().st_size}')
    print(f'candidate_pairs_bytes={cand_path.stat().st_size}')
    print(f'runtime_seconds={time.time()-started:.3f}')

if __name__=='__main__': main()
