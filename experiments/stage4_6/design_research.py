#!/usr/bin/env python3
"""Stage 4.6: candidate concentration and rule overlap research (training only)."""
import csv, hashlib, json, random, statistics, sys, time
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'student_resource/dataset/train'; OUT=ROOT/'experiments/stage4_6'
sys.path.insert(0,str(ROOT/'experiments/stage4_4'))
import targeted_blocking as b

SEED=20261008; STRATA=(('zero',0,0),('one',1,1),('multi',2,10**9)); COUNTRIES=('India','US')
SOURCES=(('train_source2.tsv','S2'),('train_source3.tsv','S3'))
def rows(p):
    with open(p,encoding='utf-8',newline='') as f: yield from csv.DictReader(f,delimiter='\t')
def sha(ids): return hashlib.sha256('\n'.join(ids).encode()).hexdigest()
def quant(v):
    if not v:return {'n':0,'mean':0,'median':0,'p90':0,'p95':0,'p99':0,'max':0,'total':0}
    s=sorted(v); return {'n':len(v),'mean':statistics.fmean(v),'median':statistics.median(s),'p90':s[(90*len(s)+99)//100-1],'p95':s[(95*len(s)+99)//100-1],'p99':s[(99*len(s)+99)//100-1],'max':s[-1],'total':sum(v)}
def sample():
    # Read the already saved Stage 4.3-4.5 samples for explicit exclusions.
    old=set()
    for folder,files in [('stage4_3',['sample_ids.txt']),('stage4_4',['sample_ids.txt']),('stage4_5',['sample1_ids.txt','sample2_ids.txt','sample3_ids.txt'])]:
        for fn in files: old.update((ROOT/'experiments'/folder/fn).read_text().splitlines())
    gt={}
    for r in rows(DATA/'train_ground_truth.tsv'):
        raw=r['matched_entity_ids'].strip(); links=[x.strip() for x in raw.split(',') if x.strip()] if raw else []
        gt[r['source1_entity_id']]=links
    pools={(cl,c):[] for cl,_,_ in STRATA for c in COUNTRIES}
    for r in rows(DATA/'train_source1.tsv'):
        sid=r['entity_id']; links=gt[sid]; cl='zero' if not links else 'one' if len(links)==1 else 'multi'; c=r['country']
        if c in COUNTRIES and sid not in old and hashlib.sha256(sid.encode()).digest()[0]>=64:pools[(cl,c)].append((sid,r,links))
    rng=random.Random(SEED); selected=[]
    for cl,_,_ in STRATA:
        for c in COUNTRIES:
            pool=sorted(pools[(cl,c)],key=lambda x:x[0]); selected.extend(rng.sample(pool,50))
    order={(cl,c):(i,j) for i,(cl,_,_) in enumerate(STRATA) for j,c in enumerate(COUNTRIES)}
    selected.sort(key=lambda x:(order[('zero' if not x[2] else 'one' if len(x[2])==1 else 'multi',x[1]['country'])],x[0]))
    out=[]
    for sid,r,links in selected:
        out.append({'id':sid,'country':r['country'],'name':r['business_name'],'address':r['business_address'],'links':links,'class':'zero' if not links else 'one' if len(links)==1 else 'multi'})
    spec={'seed':SEED,'n':len(out),'class_country_quota':50,'class_counts':dict(Counter(x['class'] for x in out)),'country_counts':dict(Counter(x['country'] for x in out)),'true_links':sum(map(lambda x:len(x['links']),out)),'s2_true_links':sum(t.startswith('S2-') for x in out for t in x['links']),'s3_true_links':sum(t.startswith('S3-') for x in out for t in x['links']),'sha256_ordered_ids':sha([x['id'] for x in out]),'hash_input':'UTF-8 IDs LF-joined without trailing LF','sampling':'sorted eligible IDs in each class×country pool; one stdlib random.Random(20261008), sample 50 without replacement per stratum; zero/one/multi then India/US; final IDs sorted class, country, ID','eligibility':'India/US, classes 0/1/>=2, SHA256(ID) first byte >=64, excluding saved Stage 4.3, 4.4, and all Stage 4.5 IDs'}
    if len(out)!=300 or len(old & {x['id'] for x in out}):raise ValueError('sample failed')
    (OUT/'sample_ids.txt').write_text('\n'.join(x['id'] for x in out)+'\n')
    return out,spec
def enrich(q):
    for x in q:
        x['prefix']=b.normalize_name(x['name']) or ''; x['p5']=x['prefix'][:5] if len(x['prefix'])>=5 else ''
        x['a']=b.address_keys(x['address']); x['n']=b.name_tokens(x['name'])
def run(tag):
    start=time.time(); q,spec=sample(); enrich(q)
    qprefix={(x['country'],x['p5']) for x in q if x['p5']}; qa={(x['country'],t) for x in q for t in x['a']}; qn={(x['country'],t) for x in q for t in x['n']}
    dfs={src:{'a':Counter(),'n':Counter(),'p':Counter()} for _,src in SOURCES}
    for fn,src in SOURCES:
        for r in rows(DATA/fn):
            c=r['country']; _,aa,nn=b.target_keys(r)
            if (c,(b.normalize_name(r['business_name']) or '')[:5]) in qprefix and len(b.normalize_name(r['business_name']) or '')>=5:dfs[src]['p'][(c,(b.normalize_name(r['business_name']) or '')[:5])]+=1
            for t in aa:
                if (c,t) in qa:dfs[src]['a'][(c,t)]+=1
            for t in nn:
                if (c,t) in qn:dfs[src]['n'][(c,t)]+=1
    freq={k:Counter() for k in ('a','n','p')}
    for src in ('S2','S3'):
        for k in freq:freq[k].update(dfs[src][k])
    # Retain only queried postings that are eligible under unchanged caps.
    post={src:{k:defaultdict(set) for k in ('a','n','p')} for src in ('S2','S3')}
    for fn,src in SOURCES:
        for r in rows(DATA/fn):
            tid=r['entity_id']; c=r['country']; nn=b.normalize_name(r['business_name']) or ''; _,aa,nt=b.target_keys(r)
            pk=(c,nn[:5])
            if len(nn)>=5 and pk in qprefix:post[src]['p'][pk].add(tid)
            for t in aa:
                key=(c,t)
                if key in qa and freq['a'][key]<=25000:post[src]['a'][key].add(tid)
            for t in nt:
                key=(c,t)
                if key in qn and freq['n'][key]<=25000:post[src]['n'][key].add(tid)
    per={}
    passes=('prefix5','address_le2500','address_2501_25000','name_le2500','hf_address_and_name')
    for src in ('S2','S3'):
        per[src]={p:[] for p in passes}; recovered={p:0 for p in ('A','B','A_HF_NAME','A_NAME2500')}; truth={t for x in q for t in x['links'] if t.startswith(src+'-')}
        unions={p:0 for p in passes}; key_raw=Counter(); key_unique=Counter(); key_source=Counter()
        for x in q:
            c=x['country']; p5=post[src]['p'].get((c,x['p5']),set()) if x['p5'] else set(); low=set(); high=set(); name=set(); name_low=set(); keys_by_type={'prefix5':[(c,x['p5'])] if x['p5'] else [],'address_le2500':[],'address_2501_25000':[],'name_le2500':[]}
            for t in x['a']:
                key=(c,t); df=freq['a'][key]
                if 0<df<=2500:
                    ids=post[src]['a'].get(key,set()); low.update(ids); keys_by_type['address_le2500'].append((key,ids))
                elif 2500<df<=25000:high.update(post[src]['a'].get(key,set())); keys_by_type['address_2501_25000'].append((key,post[src]['a'].get(key,set())))
            for t in x['n']:
                key=(c,t); df=freq['n'][key]
                if 0<df<=25000:name.update(post[src]['n'].get(key,set()))
                if 0<df<=2500:
                    ids=post[src]['n'].get(key,set()); name_low.update(ids); keys_by_type['name_le2500'].append((key,ids))
            hf=high & name; aset=p5|low; bset=p5|low|high; sets={'prefix5':p5,'address_le2500':low,'address_2501_25000':high,'name_le2500':name_low,'hf_address_and_name':hf}
            rules={'A':aset,'B':bset,'A_HF_NAME':aset|hf,'A_NAME2500':aset|name_low}
            for rule,ids in rules.items():recovered[rule]+=len(ids & set(x['links']))
            for p,ids in sets.items():per[src][p].append(len(ids)); unions[p]+=len(ids)
            keys_by_type['prefix5']=[((c,x['p5']),p5)] if x['p5'] else []
            for p,items in keys_by_type.items():
                seen=set()
                for key,ids in items:
                    key_raw[(p,key)]+=len(ids); key_unique[(p,key)]+=len(ids-seen); seen.update(ids)
            # Key-level high frequency contribution to eligible address/name raw postings.
            for t in x['a']:
                key=(c,t); df=freq['a'][key]
                if 0<df<=25000:key_source[('address_hf' if df>2500 else 'address_low',key)]+=len(post[src]['a'].get(key,set()))
        per[src]['rule_recall']={k:{'recovered':v,'true_links':len(truth),'recall':v/len(truth) if truth else None} for k,v in recovered.items()}
        per[src]['candidate_volume']={k:quant(v) for k,v in per[src].items() if isinstance(v,list)}
        per[src]['source_total_candidate_pairs_by_pass']=unions
        per[src]['source_true_links']=len(truth)
    # Aggregate exact per-S1 overlap and concentration, using the union of S2/S3 IDs.
    overlap={k:[] for k in ('prefix5_vs_address_le2500_jaccard','A_vs_HF_recovery_jaccard','A_vs_name_recovery_jaccard')}; rule_counts={k:[] for k in ('A','B','A_HF_NAME','A_NAME2500')}; largest_share=[]; union_sets={}; missed={k:0 for k in ('A','B','A_HF_NAME','A_NAME2500')}
    for x in q:
        c=x['country']; groups={r:set() for r in ('p','al','ah','nl','nh')}
        for src in ('S2','S3'):
            if x['p5']:groups['p']|=post[src]['p'].get((c,x['p5']),set())
            for t in x['a']:
                key=(c,t); df=freq['a'][key]
                if 0<df<=2500:groups['al']|=post[src]['a'].get(key,set())
                elif df<=25000:groups['ah']|=post[src]['a'].get(key,set())
            for t in x['n']:
                key=(c,t); df=freq['n'][key]
                if 0<df<=2500:groups['nl']|=post[src]['n'].get(key,set())
                elif df<=25000:groups['nh']|=post[src]['n'].get(key,set())
        a=groups['p']|groups['al']; bset=a|groups['ah']; hf=a|(groups['ah']&groups['nh']); name=a|groups['nl']; sets={'A':a,'B':bset,'A_HF_NAME':hf,'A_NAME2500':name}
        for k,v in sets.items():rule_counts[k].append(len(v)); missed[k]+=sum(t not in v for t in x['links'])
        for k,u,v in [('prefix5_vs_address_le2500_jaccard',groups['p'],groups['al']),('A_vs_HF_recovery_jaccard',a,groups['ah']&groups['nh']),('A_vs_name_recovery_jaccard',a,groups['nl'])]:overlap[k].append(len(u&v)/len(u|v) if u|v else 1.0)
        # Largest single-key raw posting share, calculated over query keys in A's passes.
        contributions=[]
        for typ,tokens in [('p',[x['p5']] if x['p5'] else []),('a',x['a'])]:
            for tok in tokens:
                key=(c,tok); df=freq['p' if typ=='p' else 'a'][key]
                if typ=='p' or 0<df<=2500:
                    contributions.append(sum(len(post[s]['p' if typ=='p' else 'a'].get(key,set())) for s in ('S2','S3')))
        largest_share.append(max(contributions,default=0)/max(1,sum(contributions)))
    # Posting-list frequency distribution and top keys, target-source splits retained.
    key_stats={}
    for typ in ('a','n','p'):
        vals=list(freq[typ].values()); key_stats[typ]={'queried_keys':len(vals),'df_distribution':quant(vals),'top_keys':[{'country':k[0],'key':k[1],'df':v,'S2_df':dfs['S2'][typ][k],'S3_df':dfs['S3'][typ][k]} for k,v in freq[typ].most_common(30)],'summed_df_by_band':{band:sum(v for v in vals if lo<=v<=hi) for band,lo,hi in [('1-10',1,10),('11-100',11,100),('101-2500',101,2500),('2501-25000',2501,25000),('>25000',25001,10**12)]}}
    result={'title':'Stage 4.6 candidate-generator design research','run_tag':tag,'sample':spec,'definitions_preserved':{'A':'country + normalized name prefix-5 UNION address tokens with combined target df 1..2500','B':'same prefix-5 UNION address tokens with combined target df 1..25000','A_HF_NAME':'A UNION (shared address df 2501..25000 AND shared name token df 1..25000)','A_NAME2500':'A UNION shared name token df 1..2500','source-specific volume':'source-scoped candidate target IDs; combined frequency caps are computed over S2+S3 as before'},'posting_frequency':key_stats,'per_source':per,'combined_per_s1':{'candidate_count_distributions':{k:quant(v) for k,v in rule_counts.items()},'candidate_misses':missed,'pass_overlap_mean_jaccard':{k:statistics.fmean(v) if v else 0 for k,v in overlap.items()},'largest_A_single_key_share':quant(largest_share)},'limitation':'Posting summaries describe only keys queried by this deterministic balanced sample; they are not global source-wide key rankings. No candidate output matrix is persisted.','runtime_seconds':round(time.time()-start,2)}
    return result
def main():
    OUT.mkdir(parents=True,exist_ok=True); a=run('run1'); (OUT/'run1.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n'); bres=run('run2'); (OUT/'run2.json').write_text(json.dumps(bres,ensure_ascii=False,indent=2)+'\n')
    def clean(o):
        if isinstance(o,dict):return {k:clean(v) for k,v in o.items() if k not in ('runtime_seconds','run_tag')}
        if isinstance(o,list):return [clean(v) for v in o]
        return o
    same=clean(a)==clean(bres); (OUT/'reproducibility.json').write_text(json.dumps({'passed':same,'exact_results_equal_except_runtime_and_tag':same,'sample_sha256':bres['sample']['sha256_ordered_ids']},indent=2)+'\n')
    if not same:raise RuntimeError('Stage 4.6 rerun mismatch')
    print(json.dumps({'passed':same,'runtime_run1':a['runtime_seconds'],'runtime_run2':bres['runtime_seconds'],'sample_sha256':bres['sample']['sha256_ordered_ids']}))
if __name__=='__main__':main()
