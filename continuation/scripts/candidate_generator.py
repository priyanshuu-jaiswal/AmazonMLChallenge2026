"""Frozen Stage 4.7 A+HF-address/name candidate generation."""
from __future__ import annotations
import csv
from collections import Counter,defaultdict
from pathlib import Path
from normalization.normalize import normalize_name
from experiments.stage4_4.targeted_blocking import address_keys,name_tokens,target_keys

def read_rows(path):
    with open(path,encoding='utf-8',newline='') as f:yield from csv.DictReader(f,delimiter='\t')
def qkeys(s1):
    n=normalize_name(s1['business_name']) or ''
    return ( (s1['country'],n[:5]) if len(n)>=5 else None,
             {(s1['country'],t) for t in address_keys(s1.get('business_address'))},
             {(s1['country'],t) for t in name_tokens(s1['business_name'])})
def build_index(s1_rows,target_paths,preserve_ids=()):
    preserve_ids=set(preserve_ids)
    qs=[qkeys(x) for x in s1_rows]
    qp={x[0] for x in qs if x[0]}; qa=set().union(*(x[1] for x in qs)); qn=set().union(*(x[2] for x in qs))
    dfp=Counter(); dfa=Counter(); dfn=Counter(); target_info={}
    for path in target_paths:
        for r in read_rows(path):
            p,a,n=target_keys(r); c=r['country']
            if 5 in p and (c,p[5][1]) in qp:dfp[(c,p[5][1])]+=1
            for t in a:
                if (c,t) in qa:dfa[(c,t)]+=1
            for t in n:
                if (c,t) in qn:dfn[(c,t)]+=1
            if r['entity_id'] in preserve_ids:target_info[r['entity_id']]=r
    ppost=defaultdict(list); apost=defaultdict(list); npost=defaultdict(list)
    for path in target_paths:
        for r in read_rows(path):
            tid=r['entity_id']; c=r['country']; p,a,n=target_keys(r); keep=False
            if 5 in p:
                key=(c,p[5][1])
                if key in qp:ppost[key].append(tid);keep=True
            for t in a:
                key=(c,t)
                if key in qa and 0<dfa[key]<=25000:apost[key].append(tid);keep=True
            for t in n:
                key=(c,t)
                if key in qn and 0<dfn[key]<=25000:npost[key].append(tid);keep=True
            if keep:target_info[tid]=r
    return {'queries':qs,'dfp':dfp,'dfa':dfa,'dfn':dfn,'ppost':ppost,'apost':apost,'npost':npost,'target_info':target_info}
def candidates(s1,index):
    pk,ak,nk=qkeys(s1); c=s1['country']; pref=set(index['ppost'].get(pk,())) if pk else set()
    low=set(); high=Counter()
    for key in ak:
        df=index['dfa'].get(key,0); ids=index['apost'].get(key,())
        if 0<df<=2500:low.update(ids)
        elif 2500<df<=25000:high.update(ids)
    names=set()
    for key in nk:
        if 0<index['dfn'].get(key,0)<=25000:names.update(index['npost'].get(key,()))
    a=pref|low
    return a|(set(high)&names)
