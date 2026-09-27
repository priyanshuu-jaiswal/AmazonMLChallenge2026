#!/usr/bin/env python3
"""Entity-level held-out validation of the interpretable Stage 5 scorer."""
import csv,hashlib,heapq,json,sqlite3,tempfile,sys,time,statistics,difflib,unicodedata
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'student_resource/dataset/train'; OUT=ROOT/'experiments/stage5'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/stage5'))
from candidate_generator import read_rows,build_index,candidates
from features import pair_features
from scorer import score,WEIGHTS
SPLIT_SEED=20261011; VAL_SEED=20261012; N_PER_STRATUM=50
CLASSES=('zero','one','multi'); THRESHOLDS=[round(5+i*.25,2) for i in range(49)]
def link_list(raw):return [x.strip() for x in raw.split(',') if x.strip()] if raw.strip() else []
def sha(ids):return hashlib.sha256('\n'.join(ids).encode()).hexdigest()
def scriptset(s):return {unicodedata.name(c,'').split()[0] for c in s if c.isalpha()}
def select_validation():
    with tempfile.NamedTemporaryFile(prefix='stage5_split_',suffix='.sqlite') as tf:
        db=sqlite3.connect(tf.name);db.execute('PRAGMA journal_mode=OFF');db.execute('PRAGMA synchronous=OFF')
        db.execute('CREATE TABLE s(id TEXT PRIMARY KEY,c TEXT,nm TEXT,ad TEXT)');db.execute('CREATE TABLE g(id TEXT PRIMARY KEY,links TEXT,n INTEGER)')
        for i,r in enumerate(read_rows(DATA/'train_source1.tsv'),1):
            db.execute('INSERT INTO s VALUES(?,?,?,?)',(r['entity_id'],r['country'],r['business_name'],r['business_address']))
            if i%100000==0:db.commit()
        db.commit()
        for i,r in enumerate(read_rows(DATA/'train_ground_truth.tsv'),1):
            ls=link_list(r['matched_entity_ids']);db.execute('INSERT INTO g VALUES(?,?,?)',(r['source1_entity_id'],json.dumps(ls),len(ls)))
            if i%100000==0:db.commit()
        db.commit(); strata=Counter(); heaps={};
        for sid,c,n in db.execute('SELECT s.id,s.c,g.n FROM s JOIN g ON s.id=g.id'):
            cl='zero' if n==0 else 'one' if n==1 else 'multi'; bucket=hashlib.sha256(f'{SPLIT_SEED}\0{sid}'.encode()).digest()[0]%5
            if bucket!=0:continue
            key=(cl,c);strata[key]+=1;heaps.setdefault(key,[]);rank=int.from_bytes(hashlib.sha256(f'{VAL_SEED}\0{sid}'.encode()).digest(),'big');item=(-rank,sid)
            if len(heaps[key])<N_PER_STRATUM:heapq.heappush(heaps[key],item)
            elif rank < -heaps[key][0][0]:heapq.heapreplace(heaps[key],item)
        if any(len(h)<N_PER_STRATUM for h in heaps.values()):raise RuntimeError('Insufficient validation stratum')
        countries=sorted({c for _,c in heaps});
        if not {'India','US'}.issubset(countries):raise RuntimeError(f'Missing required validation country: {countries}')
        chosen=[sid for h in heaps.values() for _,sid in h]; records=[]
        for sid in chosen:
            c,nm,ad=db.execute('SELECT c,nm,ad FROM s WHERE id=?',(sid,)).fetchone();lj,n=db.execute('SELECT links,n FROM g WHERE id=?',(sid,)).fetchone();ls=json.loads(lj)
            records.append({'entity_id':sid,'country':c,'business_name':nm,'business_address':ad,'truth':ls,'class':'zero' if n==0 else 'one' if n==1 else 'multi'})
        db.close()
    order={'zero':0,'one':1,'multi':2};records.sort(key=lambda x:(order[x['class']],x['country'],x['entity_id']));ids=[x['entity_id'] for x in records]
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'validation_ids.txt').write_text('\n'.join(ids)+'\n')
    spec={'seed_split':SPLIT_SEED,'seed_sampling':VAL_SEED,'split':'entity-level hash split: SHA256(split_seed + NUL + S1 ID) first byte modulo 5 == 0 is held-out validation pool; other 80% reserved as development/training IDs. No fitting is done in this rule-based baseline.','sample':'50 deterministic bottom-k SHA256-ranked S1s from each match-class×country stratum in the validation pool','sample_size':len(ids),'sample_sha256':sha(ids),'strata':{f'{c}/{k}':sum(x['class']==c and x['country']==k for x in records) for c in CLASSES for k in countries},'validation_pool_counts':{f'{c}/{k}':strata[(c,k)] for c in CLASSES for k in countries},'countries':countries,'true_links':sum(len(x['truth']) for x in records),'s2_true_links':sum(t.startswith('S2-') for x in records for t in x['truth']),'s3_true_links':sum(t.startswith('S3-') for x in records for t in x['truth'])}
    return records,spec
def f05(p,r):return 1.25*p*r/(.25*p+r) if .25*p+r else 0.0
def metrics_for(rows,threshold):
    per=[];tp=fp=fn=pred=0;zero_n=zero_correct=0; details=[]
    for row in rows:
        truth=set(row['truth']);preds={tid for tid,s in row['scored'] if s>=threshold};hit=len(truth&preds);falsep=len(preds-truth);falsen=len(truth-preds)
        if not truth and not preds: p=r=f=1.0
        else:
            p=hit/len(preds) if preds else 0.0;r=hit/len(truth) if truth else 0.0;f=f05(p,r)
        per.append(f);tp+=hit;fp+=falsep;fn+=falsen;pred+=len(preds)
        if not truth:zero_n+=1;zero_correct+=int(not preds)
        details.append({'s1_id':row['entity_id'],'class':row['class'],'truth':sorted(truth),'predicted':sorted(preds),'tp':hit,'fp':falsep,'fn':falsen,'f05':f})
    return {'threshold':threshold,'macro_f0_5':statistics.fmean(per),'micro_precision':tp/(tp+fp) if tp+fp else 0.0,'micro_recall':tp/(tp+fn) if tp+fn else 0.0,'true_positive_links':tp,'false_positive_count':fp,'false_negative_count':fn,'predicted_match_count':pred,'zero_match_s1':zero_n,'zero_match_correct_empty':zero_correct,'zero_match_accuracy':zero_correct/zero_n if zero_n else None},details
def main():
    start=time.time();val,spec=select_validation();truth_targets={t for x in val for t in x['truth']}
    idx=build_index(val,[DATA/'train_source2.tsv',DATA/'train_source3.tsv'],preserve_ids=truth_targets)
    for x in val:
        ids=candidates(x,idx); missing=ids-set(idx['target_info'])
        if missing:raise RuntimeError(f'Missing target features for {len(missing)} candidate IDs')
        scored=[]
        for tid in ids:
            ft=pair_features(x,idx['target_info'][tid]);scored.append((tid,score(ft)))
        x['scored']=scored
    sweep=[];all_details=[]
    for th in THRESHOLDS:
        m,d=metrics_for(val,th);sweep.append(m);all_details.append((th,d))
    chosen=max(sweep,key=lambda m:(m['macro_f0_5'],m['threshold']))['threshold']
    ranked=sorted(sweep,key=lambda m:(m['macro_f0_5'],m['threshold']),reverse=True)[:3]
    top_errors=[]
    for th in [x['threshold'] for x in ranked]:
        _,ds=metrics_for(val,th);by={x['entity_id']:x for x in val};examples=[]
        for d in ds:
            if d['fp'] or d['fn']:
                x=by[d['s1_id']];errors=[]
                for tid in d['predicted']:
                    if tid not in d['truth']:
                        t=idx['target_info'][tid];sn=__import__('normalization.normalize',fromlist=['normalize_name']).normalize_name(x['business_name']) or '';tn=__import__('normalization.normalize',fromlist=['normalize_name']).normalize_name(t['business_name']) or ''
                        cats=[]
                        if not (t.get('business_address') or '').strip():cats.append('missing_address')
                        if scriptset(sn)!=scriptset(tn):cats.append('multilingual_script')
                        if set(sn.split())==set(tn.split()) and sn!=tn:cats.append('name_order')
                        if sn!=tn and difflib.SequenceMatcher(None,sn,tn).ratio()>=.72:cats.append('typo_like')
                        if not cats:cats.append('other')
                        errors.append({'type':'false_positive','target_id':tid,'categories':cats,'s1_name':x['business_name'],'target_name':t['business_name']})
                for tid in d['truth']:
                    if tid not in d['predicted']:
                        t=idx['target_info'].get(tid,{}) ; sn=__import__('normalization.normalize',fromlist=['normalize_name']).normalize_name(x['business_name']) or '';tn=__import__('normalization.normalize',fromlist=['normalize_name']).normalize_name(t.get('business_name','')) or '';cats=[]
                        if not (t.get('business_address') or '').strip():cats.append('missing_address')
                        if scriptset(sn)!=scriptset(tn):cats.append('multilingual_script')
                        if set(sn.split())==set(tn.split()) and sn!=tn:cats.append('name_order')
                        if sn!=tn and difflib.SequenceMatcher(None,sn,tn).ratio()>=.72:cats.append('typo_like')
                        if not cats:cats.append('other')
                        errors.append({'type':'false_negative','target_id':tid,'categories':cats,'s1_name':x['business_name'],'target_name':t.get('business_name','')})
                examples.append({'s1_id':d['s1_id'],'match_class':d['class'],'zero_match_false_positive':d['class']=='zero' and d['fp']>0,'missed_singleton':d['class']=='one' and d['fn']>0,'missed_multi_links':d['class']=='multi' and d['fn']>0,'errors':errors[:10]})
        top_errors.append({'threshold':th,'metrics':next(m for m in sweep if m['threshold']==th),'error_entities':examples[:30]})
    chosen_row=next(m for m in sweep if m['threshold']==chosen)
    # candidate recall ceiling, before score threshold
    cand_links=sum(len(x['truth']) for x in val); cand_recovered=sum(tid in {i for i,_ in x['scored']} for x in val for tid in x['truth'])
    result={'title':'Stage 5 rule-based validation baseline','validation_sample':spec,'scorer_weights':WEIGHTS,'scorer_formula':'sum(weight × feature); country exact bonus, address missingness penalty; threshold applied to each candidate independently','threshold_sweep':sweep,'chosen_threshold':chosen,'chosen_metrics':chosen_row,'candidate_recall_ceiling':{'recovered':cand_recovered,'true':cand_links,'recall':cand_recovered/cand_links if cand_links else None},'strongest_threshold_error_analysis':top_errors,'runtime_seconds':round(time.time()-start,3)}
    (OUT/'threshold_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'sample':spec,'threshold':chosen,'metrics':chosen_row,'candidate_recall_ceiling':result['candidate_recall_ceiling'],'runtime_seconds':result['runtime_seconds']},indent=2))
if __name__=='__main__':main()
