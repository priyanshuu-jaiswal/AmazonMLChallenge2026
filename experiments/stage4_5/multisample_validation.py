#!/usr/bin/env python3
"""Stage 4.5 multi-sample blocking validation; supplied training data only."""
import csv, hashlib, json, random, sqlite3, tempfile, sys, time, statistics
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'student_resource/dataset/train'; OUT=ROOT/'experiments/stage4_5'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'experiments/stage4_4'))
import targeted_blocking as b

SEEDS=(20261005,20261006,20261007)
CLASSES=(('zero',0,0),('one',1,1),('multi',2,10**9)); COUNTRIES=('India','US')
RULES=('A','B','A_HF_NAME','A_NAME2500')

def rows(path):
    with open(path,encoding='utf-8',newline='') as f: yield from csv.DictReader(f,delimiter='\t')
def shalines(v): return hashlib.sha256('\n'.join(v).encode()).hexdigest()

def baseline_gate():
    s43=json.load(open(ROOT/'experiments/stage4_3/latest_results.json'))
    s44=json.load(open(ROOT/'experiments/stage4_4/run1.json'))
    audit=json.load(open(ROOT/'experiments/stage4_4/reproducibility.json'))
    if not audit['passed']: raise RuntimeError('Stage 4.4 historical reproducibility audit failed')
    want43={'Union cap 2,500':(460,487),'Union cap 25,000':(484,487)}
    for k,(rec,tru) in want43.items():
        z=s43['strategies'][k]
        if (z['recovered_links'],z['true_links'])!=(rec,tru): raise RuntimeError(f'Stage 4.3 baseline mismatch {k}')
    want44={'A':(476,491),'B':(488,491),'A_HF_NAME':(485,491),'A_NAME2500':(485,491)}
    for k,(rec,tru) in want44.items():
        z=s44['new_validation']['strategies'][k]
        if (z['recovered_links'],z['true_links'])!=(rec,tru): raise RuntimeError(f'Stage 4.4 baseline mismatch {k}')
    return {'stage43':{k:{'recovered':v[0],'true':v[1]} for k,v in want43.items()},
            'stage44':{k:{'recovered':v[0],'true':v[1]} for k,v in want44.items()},'stage44_reproducibility_passed':True}

def sampling_pass():
    # Streaming source tables hold only selection fields and labels, never target data.
    with tempfile.NamedTemporaryFile(prefix='s45_',suffix='.sqlite') as f:
        db=sqlite3.connect(f.name); db.execute('PRAGMA journal_mode=OFF'); db.execute('PRAGMA synchronous=OFF')
        db.execute('CREATE TABLE s(id TEXT PRIMARY KEY,c TEXT,nm TEXT,ad TEXT)')
        db.execute('CREATE TABLE g(id TEXT PRIMARY KEY,links TEXT,n INTEGER)')
        for i,r in enumerate(rows(DATA/'train_source1.tsv'),1):
            db.execute('INSERT INTO s VALUES(?,?,?,?)',(r['entity_id'],r['country'],r['business_name'],r['business_address']))
            if i%100000==0: db.commit()
        db.commit()
        for i,r in enumerate(rows(DATA/'train_ground_truth.tsv'),1):
            links=[x.strip() for x in r['matched_entity_ids'].split(',') if x.strip()] if r['matched_entity_ids'].strip() else []
            db.execute('INSERT INTO g VALUES(?,?,?)',(r['source1_entity_id'],json.dumps(links),len(links)))
            if i%100000==0: db.commit()
        db.commit(); old=set((ROOT/'experiments/stage4_3/sample_ids.txt').read_text().splitlines())|set((ROOT/'experiments/stage4_4/sample_ids.txt').read_text().splitlines())
        selected=[]; excluded=set(old); specs=[]
        for ix,seed in enumerate(SEEDS,1):
            rng=random.Random(seed); sample=[]
            for cls,lo,hi in CLASSES:
                for country in COUNTRIES:
                    pool=[z[0] for z in db.execute('SELECT s.id FROM s JOIN g ON s.id=g.id WHERE s.c=? AND g.n BETWEEN ? AND ? ORDER BY s.id',(country,lo,hi)) if hashlib.sha256(z[0].encode()).digest()[0]>=64 and z[0] not in excluded]
                    if len(pool)<50: raise RuntimeError(f'Insufficient eligible stratum {ix}/{cls}/{country}: {len(pool)}')
                    for sid in rng.sample(pool,50):
                        c,nm,ad=db.execute('SELECT c,nm,ad FROM s WHERE id=?',(sid,)).fetchone(); lj,n=db.execute('SELECT links,n FROM g WHERE id=?',(sid,)).fetchone()
                        sample.append({'id':sid,'country':c,'name':nm,'address':ad,'links':json.loads(lj),'match_count':n,'class':cls})
            co={x:i for i,(x,_,_) in enumerate(CLASSES)}; cc={x:i for i,x in enumerate(COUNTRIES)}
            sample.sort(key=lambda x:(co[x['class']],cc[x['country']],x['id']))
            ids=[x['id'] for x in sample]
            if len(set(ids))!=300 or set(ids)&excluded: raise RuntimeError('Sample overlap/cardinality failure')
            # Fresh RNG and re-created ordered eligible pools independently replay this draw.
            rr=random.Random(seed); replay=[]
            for cls,lo,hi in CLASSES:
                for country in COUNTRIES:
                    pool=[z[0] for z in db.execute('SELECT s.id FROM s JOIN g ON s.id=g.id WHERE s.c=? AND g.n BETWEEN ? AND ? ORDER BY s.id',(country,lo,hi)) if hashlib.sha256(z[0].encode()).digest()[0]>=64 and z[0] not in excluded]
                    replay.extend(rr.sample(pool,50))
            replay.sort(key=lambda sid:(co[next(x['class'] for x in sample if x['id']==sid)],cc[next(x['country'] for x in sample if x['id']==sid)],sid))
            if replay!=ids: raise RuntimeError('Independent sample replay failed')
            classes=Counter(x['class'] for x in sample); countries=Counter(x['country'] for x in sample)
            links=[t for x in sample for t in x['links']]; s2=sum(t.startswith('S2-') for t in links); s3=sum(t.startswith('S3-') for t in links)
            if (len(sample),classes['zero'],classes['one'],classes['multi'],countries['India'],countries['US'],len(links))!=(300,100,100,100,150,150,len(links)) or len(links)!=s2+s3: raise RuntimeError('Sample stratum/link accounting failed')
            spec={'seed':seed,'sample_count':300,'strata':dict(classes),'country_counts':dict(countries),'strata_country_counts':{f'{cl}/{c}':sum(x['class']==cl and x['country']==c for x in sample) for cl,_,_ in CLASSES for c in COUNTRIES},'true_links':len(links),'s2_true_links':s2,'s3_true_links':s3,'sample_sha256':shalines(ids),'hash_input':'UTF-8 ordered S1 IDs joined by LF, no trailing LF','source_order':'S1 IDs lexicographically sorted within exact class × country pools','filtering':'join supplied train_source1 and train_ground_truth; India/US; 0/1/>=2 links; SHA256(ID) first byte >=64; exclude saved Stage 4.3/4.4 IDs and all earlier Stage 4.5 samples','sampling':'stdlib random.Random(seed).sample without replacement, one RNG over zero/one/multi then India/US, 50 per stratum','final_order':'match class zero/one/multi, then India/US, then S1 ID lexicographically','replay':'passed using fresh RNG and fresh eligible-pool queries','historical_limit':'Stage 4.1/4.2 sample IDs unavailable; no claim of disjointness from those samples'}
            (OUT/f'sample{ix}_ids.txt').write_text('\n'.join(ids)+'\n')
            selected.append(sample); specs.append(spec); excluded.update(ids)
        db.close()
    return selected,specs

def enrich(sample):
    for x in sample:
        x['norm_name']=b.normalize_name(x['name']) or ''; x['prefix4']=x['norm_name'][:4] if len(x['norm_name'])>=4 else ''; x['prefix5']=x['norm_name'][:5] if len(x['norm_name'])>=5 else ''; x['prefix6']=x['norm_name'][:6] if len(x['norm_name'])>=6 else ''; x['addr_keys']=b.address_keys(x['address']); x['name_keys']=b.name_tokens(x['name'])

def run(tag):
    start=time.time(); gate=baseline_gate(); samples,specs=sampling_pass(); [enrich(s) for s in samples]
    ids={x['id'] for s in samples for x in s}; links={t for s in samples for x in s for t in x['links']}; allq=[x for s in samples for x in s]
    qp={(x['country'],k,x[f'prefix{k}']) for x in allq for k in (4,5,6) if x[f'prefix{k}']}; qa={(x['country'],t) for x in allq for t in x['addr_keys']}; qn={(x['country'],t) for x in allq for t in x['name_keys']}
    dfp=Counter(); dfa=Counter(); dfn=Counter(); targetinfo={}
    sources=(('train_source2.tsv','S2'),('train_source3.tsv','S3'))
    for fn,src in sources:
        for r in rows(DATA/fn):
            p,a,n=b.target_keys(r); c=r['country']
            for k,v in p.items():
                if (v[0],k,v[1]) in qp: dfp[(v[0],k,v[1])]+=1
            for t in a:
                if (c,t) in qa: dfa[(c,t)]+=1
            for t in n:
                if (c,t) in qn: dfn[(c,t)]+=1
            if r['entity_id'] in links: targetinfo[r['entity_id']]={'source':src,'country':c,'name':r['business_name'],'address':r['business_address'],'norm_name':b.normalize_name(r['business_name']) or '', 'addr_keys':a,'name_keys':n}
    if links-set(targetinfo): raise RuntimeError(f'Missing true target IDs: {len(links-set(targetinfo))}')
    print('frequency scan complete',flush=True)
    ppm={}; apm={}; nam={}
    from collections import defaultdict
    ppm=defaultdict(list); apm=defaultdict(list); nam=defaultdict(list)
    for fn,_ in sources:
        for r in rows(DATA/fn):
            p,a,n=b.target_keys(r); c=r['country']; tid=r['entity_id']
            for k,v in p.items():
                key=(v[0],k,v[1])
                if key in qp: ppm[key].append(tid)
            for t in a:
                key=(c,t)
                if key in qa and 0<dfa[key]<=25000: apm[key].append(tid)
            for t in n:
                key=(c,t)
                if key in qn and 0<dfn[key]<=25000: nam[key].append(tid)
    print('posting scan complete',flush=True)
    results=[]; recovered={}
    for ix,s in enumerate(samples,1):
        truth={t:('S2' if t.startswith('S2-') else 'S3') for x in s for t in x['links']}
        m,h=b.candidate_rule_results(s,ppm,apm,nam,dfp,dfa,dfn,targetinfo,truth,RULES); results.append(m); recovered[ix]=h
        # The imported Stage 4.4 evaluator reports full and zero-match volumes.
        # Calculate the matched-S1 candidate distribution with the same sets.
        matched_counts={r:[] for r in RULES}
        for x in s:
            if x['class']=='zero': continue
            c=x['country']; p5=set(ppm.get((c,5,x['prefix5']),())) if x['prefix5'] else set()
            aset=set(); lowaddr=set(); hf=Counter(); n2500=set(); nunion=set()
            for tok in x['addr_keys']:
                key=(c,tok); df=dfa.get(key,0); tids=apm.get(key,())
                if 0<df<=25000: aset.update(tids)
                if 0<df<=2500: lowaddr.update(tids)
                if 2500<df<=25000: hf.update(tids)
            for tok in x['name_keys']:
                key=(c,tok); df=dfn.get(key,0)
                if 0<df<=25000: nunion.update(nam.get(key,()))
                if 0<df<=2500: n2500.update(nam.get(key,()))
            ac=set(p5)|lowaddr
            bc=set(p5)|set(aset)
            cands={'A':ac,'B':bc,'A_HF_NAME':ac|(set(hf)&set(nunion)),'A_NAME2500':ac|n2500}
            for rule in RULES: matched_counts[rule].append(len(cands[rule]))
        for rule in RULES: m[rule]['matched_s1_candidate_volume']=b.dist(matched_counts[rule])
        # enrich candidate-purity proxies, explicitly not final precision
        total_links=len(truth)
        for rule,z in m.items():
            pairs=z['candidate_volume']['total']; z['candidate_pairs_not_ground_truth']=pairs-z['recovered_links']; z['blocking_candidate_purity_proxy']=z['recovered_links']/pairs if pairs else None; z['candidate_pairs_per_true_link']=pairs/total_links if total_links else None
        # confirm every sampled truth ID exists in corresponding target table by membership in streamed target info.
        if set(truth)-set(targetinfo): raise RuntimeError('Target existence check failed')
        print(f'sample {ix} evaluated',flush=True)
    return {'title':'Stage 4.5 multi-sample blocking validation','run_tag':tag,'baseline_gate':gate,'samples':[{'specification':specs[i],'strategies':results[i]} for i in range(3)],'definitions':{'A':'country + conservative normalized name prefix-5 UNION country + distinct conservative/approved-abbreviation address tokens with target df 1..2500','B':'same, address df 1..25000','A_HF_NAME':'A UNION targets sharing country-address token df 2501..25000 AND any shared country-name token df 1..25000','A_NAME2500':'A UNION targets sharing any country-name token df 1..2500','candidate_count':'unique S2+S3 target IDs per S1','quantile':'nearest rank ceil(p*n)','purity':'blocking-level true candidate links / candidate pairs; not final matching precision'},'runtime_seconds':round(time.time()-start,3)}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    a=run('run1'); (OUT/'run1.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
    bres=run('run2'); (OUT/'run2.json').write_text(json.dumps(bres,ensure_ascii=False,indent=2)+'\n')
    def strip(v):
        if isinstance(v,dict): return {k:strip(x) for k,x in v.items() if k not in ('runtime_seconds','run_tag')}
        if isinstance(v,list): return [strip(x) for x in v]
        return v
    passed=strip(a)==strip(bres)
    audit={'passed':passed,'sample_hashes':[x['specification']['sample_sha256'] for x in bres['samples']],'samples_compared':3,'all_results_exactly_equal_except_runtime_and_tag':passed}
    (OUT/'reproducibility.json').write_text(json.dumps(audit,indent=2)+'\n')
    if not passed: raise RuntimeError('Stage 4.5 full rerun mismatch')
    print(json.dumps(audit,indent=2))
if __name__=='__main__': main()
