#!/usr/bin/env python3
"""Stage 4.7 final population-weighted validation of the four frozen blockers."""
from __future__ import annotations
import csv, hashlib, heapq, json, random, sqlite3, statistics, sys, tempfile, time, difflib, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'student_resource/dataset/train'; OUT=ROOT/'experiments/stage4_7'
sys.path.insert(0,str(ROOT/'experiments/stage4_4'))
import targeted_blocking as b

SEED=20261010
SAMPLE_N=1500
CLASS_ORDER=('zero','one','multi')
CLASS_RANGE={'zero':(0,0),'one':(1,1),'multi':(2,10**9)}
SOURCES=(('train_source2.tsv','S2'),('train_source3.tsv','S3'))
RULES=('A','B','A_HF_NAME','A_NAME2500')
SLICE_NAMES=('multilingual_script','missing_target_address','name_order_variation','typo_like_name_variation')

def rows(path):
    with open(path,encoding='utf-8',newline='') as f:
        yield from csv.DictReader(f,delimiter='\t')
def sha_ids(ids): return hashlib.sha256('\n'.join(ids).encode('utf-8')).hexdigest()
def dist(vals):
    if not vals:return {'n':0,'min':0,'mean':0,'median':0,'p95':0,'p99':0,'max':0,'total':0}
    s=sorted(vals)
    def q(p): return s[max(0,(int(p*len(s)+.999999999)-1))]
    return {'n':len(s),'min':s[0],'mean':statistics.fmean(s),'median':statistics.median(s),'p95':q(.95),'p99':q(.99),'max':s[-1],'total':sum(s)}
def population_quotas(counts,n):
    total=sum(counts.values()); base={k:(n*v//total) for k,v in counts.items()}
    left=n-sum(base.values())
    order=sorted(counts,key=lambda k:(-(n*counts[k]%total),k[0],k[1]))
    for k in order[:left]:base[k]+=1
    if any(base[k]<1 for k in counts):raise RuntimeError(f'Sample size too small to represent every stratum: {base}')
    return base

def sample_data():
    excluded=set()
    for folder,files in [('stage4_3',['sample_ids.txt']),('stage4_4',['sample_ids.txt']),('stage4_5',['sample1_ids.txt','sample2_ids.txt','sample3_ids.txt']),('stage4_6',['sample_ids.txt'])]:
        for fn in files: excluded.update((ROOT/'experiments'/folder/fn).read_text(encoding='utf-8').splitlines())
    with tempfile.NamedTemporaryFile(prefix='stage47_',suffix='.sqlite') as tf:
        db=sqlite3.connect(tf.name); db.execute('PRAGMA journal_mode=OFF'); db.execute('PRAGMA synchronous=OFF'); db.execute('PRAGMA temp_store=FILE')
        db.execute('CREATE TABLE s1(id TEXT PRIMARY KEY,country TEXT,name TEXT,address TEXT)')
        db.execute('CREATE TABLE gt(id TEXT PRIMARY KEY,links TEXT NOT NULL,n INTEGER NOT NULL)')
        db.execute('CREATE TABLE excluded(id TEXT PRIMARY KEY)')
        db.executemany('INSERT INTO excluded VALUES(?)',((x,) for x in excluded))
        for i,r in enumerate(rows(DATA/'train_source1.tsv'),1):
            db.execute('INSERT INTO s1 VALUES(?,?,?,?)',(r['entity_id'],r['country'],r['business_name'],r['business_address']))
            if i%100000==0:db.commit()
        db.commit()
        for i,r in enumerate(rows(DATA/'train_ground_truth.tsv'),1):
            raw=r['matched_entity_ids'].strip(); links=[x.strip() for x in raw.split(',') if x.strip()] if raw else []
            db.execute('INSERT INTO gt VALUES(?,?,?)',(r['source1_entity_id'],json.dumps(links),len(links)))
            if i%100000==0:db.commit()
        db.commit()
        counts=Counter()
        for c,n in db.execute("SELECT s.country,g.n FROM s1 s JOIN gt g ON s.id=g.id LEFT JOIN excluded e ON s.id=e.id WHERE e.id IS NULL"):
            cl='zero' if n==0 else 'one' if n==1 else 'multi'; counts[(cl,c)]+=1
        countries=sorted({c for _,c in counts})
        if not {'India','US'}.issubset(countries):raise RuntimeError(f'Expected training-country strata not found: {countries}')
        quotas=population_quotas(counts,SAMPLE_N)
        heaps={k:[] for k in counts}
        q="SELECT s.id,s.country,g.n FROM s1 s JOIN gt g ON s.id=g.id LEFT JOIN excluded e ON s.id=e.id WHERE e.id IS NULL ORDER BY s.id"
        for sid,c,n in db.execute(q):
            cl='zero' if n==0 else 'one' if n==1 else 'multi'; key=(cl,c); qn=quotas[key]
            rank=int.from_bytes(hashlib.sha256(f'{SEED}\0{sid}'.encode()).digest(),'big')
            h=heaps[key]
            item=(-rank,sid)
            if len(h)<qn:heapq.heappush(h,item)
            elif rank < -h[0][0]:heapq.heapreplace(h,item)
        chosen={sid:key for key,h in heaps.items() for _,sid in h}
        if len(chosen)!=SAMPLE_N:raise RuntimeError('Sample size mismatch')
        order={cl:i for i,cl in enumerate(CLASS_ORDER)}
        ids=sorted(chosen,key=lambda sid:(order[chosen[sid][0]],chosen[sid][1],sid))
        sample=[]
        for sid in ids:
            c,nm,ad=db.execute('SELECT country,name,address FROM s1 WHERE id=?',(sid,)).fetchone()
            lj,n=db.execute('SELECT links,n FROM gt WHERE id=?',(sid,)).fetchone(); links=json.loads(lj)
            cl='zero' if n==0 else 'one' if n==1 else 'multi'
            sample.append({'id':sid,'country':c,'name':nm,'address':ad,'links':links,'class':cl})
        db.close()
    ids=[x['id'] for x in sample]
    if len(set(ids)&excluded):raise RuntimeError('Historical sample overlap')
    actual=Counter((x['class'],x['country']) for x in sample)
    if actual!=Counter(quotas):raise RuntimeError('Sample strata do not match population quotas')
    links=[t for x in sample for t in x['links']]
    spec={'seed':SEED,'sample_size':SAMPLE_N,'sampling_method':'population-proportional joint class×country stratified sample; deterministic bottom-k SHA-256 ranking within each stratum using SHA256(seed + NUL + UTF-8 S1 ID); no replacement','population_eligible_counts_by_stratum':{f'{k[0]}/{k[1]}':v for k,v in sorted(counts.items())},'sample_quotas_by_stratum':{f'{k[0]}/{k[1]}':v for k,v in sorted(quotas.items())},'sample_counts_by_stratum':{f'{k[0]}/{k[1]}':v for k,v in sorted(actual.items())},'excluded_known_samples':['Stage 4.3','Stage 4.4','all three Stage 4.5 samples','Stage 4.6'],'unknown_historical_samples':'Stage 4.1/4.2 exact IDs are not saved; overlap with those cannot be proven.','countries_in_training':countries,'true_links':len(links),'s2_true_links':sum(t.startswith('S2-') for t in links),'s3_true_links':sum(t.startswith('S3-') for t in links),'sha256_ordered_ids':sha_ids(ids),'hash_input':'UTF-8 final-order S1 IDs joined by LF, no trailing LF','final_order':'class zero/one/multi, country lexicographically, S1 ID lexicographically','excluded_known_id_count':len(excluded)}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'sample_ids.txt').write_text('\n'.join(ids)+'\n',encoding='utf-8')
    return sample,spec

def script_set(s):
    return {unicodedata.name(ch,'').split(' ')[0] for ch in s if ch.isalpha()}

def run(tag):
    start=time.time(); sample,spec=sample_data()
    for x in sample:
        x['norm_name']=b.normalize_name(x['name']) or ''; x['prefix5']=x['norm_name'][:5] if len(x['norm_name'])>=5 else ''; x['addr_keys']=b.address_keys(x['address']); x['name_keys']=b.name_tokens(x['name'])
    qprefix={(x['country'],x['prefix5']) for x in sample if x['prefix5']}; qaddr={(x['country'],t) for x in sample for t in x['addr_keys']}; qname={(x['country'],t) for x in sample for t in x['name_keys']}
    dfp=Counter(); dfa=Counter(); dfn=Counter(); target_info={}; alltruth={t:('S2' if t.startswith('S2-') else 'S3') for x in sample for t in x['links']}
    if any(src=='S3' and not tid.startswith('S3-') for tid,src in alltruth.items()):raise RuntimeError('Unknown ground-truth target prefix')
    for fn,src in SOURCES:
        for r in rows(DATA/fn):
            c=r['country']; pfx,aks,nts=b.target_keys(r)
            if 5 in pfx and (c,pfx[5][1]) in qprefix:dfp[(c,pfx[5][1])]+=1
            for t in aks:
                if (c,t) in qaddr:dfa[(c,t)]+=1
            for t in nts:
                if (c,t) in qname:dfn[(c,t)]+=1
            tid=r['entity_id']
            if tid in alltruth:
                target_info[tid]={'source':src,'country':c,'name':r['business_name'],'address':r['business_address'],'norm_name':b.normalize_name(r['business_name']) or '', 'addr_keys':aks,'name_keys':nts}
    if set(alltruth)!=set(target_info):raise RuntimeError(f'Missing or duplicate true target records: {len(set(alltruth)-set(target_info))}')
    print(f'{tag}: target frequency scan complete',flush=True)
    ppost=defaultdict(list); apost=defaultdict(list); npost=defaultdict(list)
    for fn,_ in SOURCES:
        for r in rows(DATA/fn):
            tid=r['entity_id']; c=r['country']; pfx,aks,nts=b.target_keys(r)
            if 5 in pfx:
                key=(c,pfx[5][1])
                if key in qprefix:ppost[key].append(tid)
            for t in aks:
                key=(c,t)
                if key in qaddr and 0<dfa[key]<=25000:apost[key].append(tid)
            for t in nts:
                key=(c,t)
                if key in qname and 0<dfn[key]<=25000:npost[key].append(tid)
    print(f'{tag}: eligible postings built',flush=True)
    cand_counts={r:[] for r in RULES}; zero_counts={r:[] for r in RULES}; source_counts={r:Counter() for r in RULES}
    source_truth={s:sum(src==s for src in alltruth.values()) for s in ('S2','S3')}
    recovered={r:set() for r in RULES}; slice_truth=Counter(); slice_recovered={r:Counter() for r in RULES}; slice_examples={c:[] for c in SLICE_NAMES}
    for x in sample:
        c=x['country']; p5=set(ppost.get((c,x['prefix5']),())) if x['prefix5'] else set(); addr_low=set();addr_all=set();addr_high=Counter();addr_target_counts=Counter()
        for t in x['addr_keys']:
            key=(c,t);df=dfa.get(key,0);ids=apost.get(key,())
            if 0<df<=2500:addr_low.update(ids);addr_all.update(ids)
            elif 2500<df<=25000:
                addr_all.update(ids); addr_high.update(ids); addr_target_counts.update(ids)
        name_all=set();name_low=set()
        for t in x['name_keys']:
            key=(c,t);df=dfn.get(key,0);ids=npost.get(key,())
            if 0<df<=25000:name_all.update(ids)
            if 0<df<=2500:name_low.update(ids)
        a=p5|addr_low; bset=p5|addr_all; hf=a|(set(addr_high)&name_all); nam=a|name_low
        candidates={'A':a,'B':bset,'A_HF_NAME':hf,'A_NAME2500':nam}
        for rule,ids in candidates.items():
            cand_counts[rule].append(len(ids)); source_counts[rule]['S2']+=sum(t.startswith('S2-') for t in ids);source_counts[rule]['S3']+=sum(t.startswith('S3-') for t in ids)
            if x['class']=='zero':zero_counts[rule].append(len(ids))
            for tid in x['links']:
                if tid in ids:recovered[rule].add((x['id'],tid))
        for tid in x['links']:
            t=target_info[tid]; sn=x['norm_name'];tn=t['norm_name']; cats=[]
            if script_set(sn)!=script_set(tn):cats.append('multilingual_script')
            if not t['address'].strip():cats.append('missing_target_address')
            if set(sn.split())==set(tn.split()) and sn!=tn:cats.append('name_order_variation')
            if sn!=tn and difflib.SequenceMatcher(None,sn,tn).ratio()>=.72:cats.append('typo_like_name_variation')
            for cat in cats:
                slice_truth[cat]+=1
                for rule in RULES:
                    if tid in candidates[rule]:slice_recovered[rule][cat]+=1
            for cat in cats:
                if len(slice_examples[cat])<25:slice_examples[cat].append({'s1_id':x['id'],'target_id':tid,'target_source':t['source'],'s1_name':x['name'],'target_name':t['name'],'scripts_s1':sorted(script_set(sn)),'scripts_target':sorted(script_set(tn)),'recovered':{r:tid in candidates[r] for r in RULES}})
    results={}
    for r in RULES:
        hits=recovered[r]; src_rec=Counter(alltruth[t] for _,t in hits)
        total=sum(cand_counts[r]); s2=source_counts[r]['S2'];s3=source_counts[r]['S3']
        if total!=s2+s3:raise RuntimeError(f'Source candidate accounting mismatch for {r}')
        if len(hits)!=src_rec['S2']+src_rec['S3']:raise RuntimeError(f'Recall accounting mismatch for {r}')
        z=zero_counts[r]; bins=Counter('0' if v==0 else '1-10' if v<=10 else '11-100' if v<=100 else '101-1000' if v<=1000 else '1001-10000' if v<=10000 else '10001-100000' if v<=100000 else '>100000' for v in z)
        results[r]={'true_links':len(alltruth),'recovered_links':len(hits),'missed_links':len(alltruth)-len(hits),'recall':len(hits)/len(alltruth),'s2':{'true':source_truth['S2'],'recovered':src_rec['S2'],'missed':source_truth['S2']-src_rec['S2'],'recall':src_rec['S2']/source_truth['S2']},'s3':{'true':source_truth['S3'],'recovered':src_rec['S3'],'missed':source_truth['S3']-src_rec['S3'],'recall':src_rec['S3']/source_truth['S3']},'candidate_pairs_total_unique':total,'candidate_pairs_s2':s2,'candidate_pairs_s3':s3,'candidate_pairs_deduplicated':total,'candidate_pairs_per_s1':dist(cand_counts[r]),'zero_match':{**dist(z),'bins':{k:bins[k] for k in ('0','1-10','11-100','101-1000','1001-10000','10001-100000','>100000')}},'recovered_link_set_sha256':hashlib.sha256('\n'.join(f'{s}\t{t}' for s,t in sorted(hits)).encode()).hexdigest(),'candidate_pairs_per_true_link':total/len(alltruth)}
    slices={cat:{'true_links':slice_truth[cat],'strategies':{r:{'recovered':slice_recovered[r][cat],'missed':slice_truth[cat]-slice_recovered[r][cat],'recall':slice_recovered[r][cat]/slice_truth[cat] if slice_truth[cat] else None} for r in RULES}} for cat in SLICE_NAMES}
    return {'title':'Stage 4.7 final candidate-generation validation','run_tag':tag,'sample':spec,'strategy_definitions':{'A':'Prefix5 UNION Address2500','B':'Prefix5 UNION Address25000','A_HF_NAME':'A UNION (shared country-address token df 2,501..25,000 AND shared country-name token df 1..25,000)','A_NAME2500':'A UNION shared country-name token df 1..2,500','normalization_and_key_definitions':'unchanged from Stage 4.4 / Stage 4.5; combined S2+S3 target df; same-country blocking'},'strategies':results,'failure_slices':slices,'failure_slice_examples':slice_examples,'runtime_seconds':round(time.time()-start,3)}

def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k not in ('runtime_seconds','run_tag')}
    if isinstance(x,list):return [clean(v) for v in x]
    return x
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    first=run('run1')
    (OUT/'results_run1.json').write_text(json.dumps(first,ensure_ascii=False,indent=2)+'\n')
    second=run('run2')
    (OUT/'results_run2.json').write_text(json.dumps(second,ensure_ascii=False,indent=2)+'\n')
    same=clean(first)==clean(second)
    audit={'passed':same,'sample_size_run1':first['sample']['sample_size'],'sample_size_run2':second['sample']['sample_size'],'sample_hash_run1':first['sample']['sha256_ordered_ids'],'sample_hash_run2':second['sample']['sha256_ordered_ids'],'sample_hashes_identical':first['sample']['sha256_ordered_ids']==second['sample']['sha256_ordered_ids'],'all_recall_candidate_volume_and_failure_slice_results_identical':same}
    (OUT/'reproducibility.json').write_text(json.dumps(audit,indent=2)+'\n')
    (OUT/'results.json').write_text(json.dumps(second,ensure_ascii=False,indent=2)+'\n')
    if not same:raise RuntimeError('Stage 4.7 full rerun mismatch')
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
