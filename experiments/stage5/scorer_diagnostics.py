#!/usr/bin/env python3
"""Reproduce the saved Stage 5 validation cohort and diagnose pair scores."""
import json, math, statistics, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/stage5'
DATA=ROOT/'student_resource/dataset/train'
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(OUT))
from validate import select_validation, f05
from candidate_generator import build_index, candidates
from features import pair_features
from scorer import score

THRESHOLDS=[4,6,8,10,12,14,16]

def distribution(vals):
    if not vals: return {'n':0}
    xs=sorted(vals)
    def pct(p): return xs[math.ceil(p*(len(xs)-1))]
    return {'n':len(xs),'min':xs[0],'p05':pct(.05),'p25':pct(.25),'median':statistics.median(xs),
            'mean':statistics.fmean(xs),'p75':pct(.75),'p95':pct(.95),'p99':pct(.99),'max':xs[-1],
            'histogram':{str(k):sum(1 for x in xs if lo<=x<hi) for k,lo,hi in
                         [(0,0,4),(4,4,6),(6,6,8),(8,8,10),(10,10,12),(12,12,14),(14,14,16),(16,16,1000)]}}

def main():
    started=time.time()
    rows,spec=select_validation()
    truth_targets={t for r in rows for t in r['truth']}
    idx=build_index(rows,[DATA/'train_source2.tsv',DATA/'train_source3.tsv'],preserve_ids=truth_targets)
    true_details=[]; false_scores=[]; cand_links=0; scored_true=0; above12=0; below12=0; total_candidates=0
    for row in rows:
        cands=candidates(row,idx); total_candidates+=len(cands); scored={}
        for tid in cands:
            ft=pair_features(row,idx['target_info'][tid]); val=score(ft); scored[tid]=(val,ft)
            if tid not in row['truth']: false_scores.append(val)
        for tid in row['truth']:
            cand_links+=1
            if tid in scored:
                scored_true+=1; val,ft=scored[tid]
                if val>=12: above12+=1
                else: below12+=1
                true_details.append({'source1_entity_id':row['entity_id'],'target_entity_id':tid,
                    'score':val,'predicted_at_12':val>=12,'features':ft})
        row['_scored']=[(tid,v[0]) for tid,v in scored.items()]
    sweep=[]
    for th in THRESHOLDS:
        per=[];tp=fp=fn=pred=0
        for row in rows:
            truth=set(row['truth']); preds={tid for tid,s in row['_scored'] if s>=th}
            hit=len(truth&preds); pfalse=len(preds-truth); missed=len(truth-preds)
            if not truth and not preds: p=r=f=1.0
            else:
                p=hit/len(preds) if preds else 0.0; r=hit/len(truth) if truth else 0.0
                f=f05(p,r)
            per.append(f);tp+=hit;fp+=pfalse;fn+=missed;pred+=len(preds)
        sweep.append({'threshold':th,'macro_f0_5':statistics.fmean(per),
                      'precision':tp/(tp+fp) if tp+fp else 0.0,'recall':tp/(tp+fn) if tp+fn else 0.0,
                      'predicted_matches':pred,'tp':tp,'fp':fp,'fn':fn})
    result={'title':'Stage 5 scorer diagnostic reproduction','validation_sample':spec,
        'candidate_true_links':cand_links,'true_links_scored':scored_true,
        'true_links_above_12':above12,'true_links_below_12':below12,
        'candidate_pairs_scored':total_candidates,
        'true_link_score_distribution':distribution([x['score'] for x in true_details]),
        'false_candidate_score_distribution':distribution(false_scores),
        'true_link_details':sorted(true_details,key=lambda x:(x['score'],x['source1_entity_id'],x['target_entity_id'])),
        'compact_threshold_sweep':sweep,'runtime_seconds':round(time.time()-started,3)}
    (OUT/'scorer_diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with open(OUT/'true_link_score_diagnostics.tsv','w',encoding='utf-8') as f:
        f.write('source1_entity_id\ttarget_entity_id\tscore\tpredicted_at_12\tfeatures_json\n')
        for x in result['true_link_details']:
            f.write(f"{x['source1_entity_id']}\t{x['target_entity_id']}\t{x['score']:.6f}\t{int(x['predicted_at_12'])}\t{json.dumps(x['features'],ensure_ascii=False,separators=(',',':'))}\n")
    print(json.dumps({k:result[k] for k in ('candidate_true_links','true_links_scored','true_links_above_12','true_links_below_12','candidate_pairs_scored','true_link_score_distribution','false_candidate_score_distribution','compact_threshold_sweep','runtime_seconds')},indent=2))

if __name__=='__main__': main()
