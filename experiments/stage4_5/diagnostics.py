#!/usr/bin/env python3
"""Deterministic diagnostics on sampled true links; no candidate-pair output."""
import csv,hashlib,json,unicodedata,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'student_resource/dataset/train'; OUT=ROOT/'experiments/stage4_5'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/stage4_4'))
import targeted_blocking as b
def rows(p):
 with open(p,encoding='utf-8',newline='') as f: yield from csv.DictReader(f,delimiter='\t')
def scripts(s): return sorted({unicodedata.name(c,'').split(' ')[0] for c in s if c.isalpha()})
def scan(tag):
 samples=[]; need=set();
 for i in range(1,4):
  ids=(OUT/f'sample{i}_ids.txt').read_text().splitlines(); samples.append({'ids':ids,'rows':[]});need.update(ids)
 s1={r['entity_id']:r for r in rows(DATA/'train_source1.tsv') if r['entity_id'] in need}
 gt={}
 for r in rows(DATA/'train_ground_truth.tsv'):
  if r['source1_entity_id'] in need:
   raw=r['matched_entity_ids'].strip();gt[r['source1_entity_id']]=[x.strip() for x in raw.split(',') if x.strip()] if raw else []
 allq=[]; trueids=set()
 for sm in samples:
  for sid in sm['ids']:
   r=s1[sid];x={'id':sid,'country':r['country'],'name':r['business_name'],'address':r['business_address'],'links':gt[sid]};x['nn']=b.normalize_name(x['name']) or '';x['prefix5']=x['nn'][:5] if len(x['nn'])>=5 else '';x['ak']=b.address_keys(x['address']);x['nk']=b.name_tokens(x['name']);sm['rows'].append(x);allq.append(x);trueids.update(x['links'])
 qaddr={(x['country'],t) for x in allq for t in x['ak']};qname={(x['country'],t) for x in allq for t in x['nk']}
 dfa=Counter();dfn=Counter();targets={}
 for fn,src in [('train_source2.tsv','S2'),('train_source3.tsv','S3')]:
  for r in rows(DATA/fn):
   c=r['country'];ak=b.address_keys(r['business_address']);nk=b.name_tokens(r['business_name'])
   for t in ak:
    if (c,t) in qaddr:dfa[(c,t)]+=1
   for t in nk:
    if (c,t) in qname:dfn[(c,t)]+=1
   if r['entity_id'] in trueids: targets[r['entity_id']]={'source':src,'country':c,'name':r['business_name'],'address':r['business_address'],'nn':b.normalize_name(r['business_name']) or '', 'ak':ak,'nk':nk}
 if trueids-set(targets):raise RuntimeError('true target missing in diagnostic scan')
 report=[]
 for ix,sm in enumerate(samples,1):
  summary={r:{'misses':0,'script_difference_misses':0,'name_mechanism_failures':0,'address_mechanism_failures':0,'both_mechanisms_fail':0,'target_address_empty_misses':0,'both_addresses_present_address_block_misses':0,'targeted_recovers_missing_address':0,'targeted_recovers_address_block_failure':0} for r in ('A','B','A_HF_NAME','A_NAME2500')}; examples={r:[] for r in summary}; cats={r:Counter() for r in summary};
  target_missing=0; both_present=0; addr_fail=0; hfrec_missing=0;namerec_missing=0;hfrec_addrfail=0;namerec_addrfail=0
  for x in sm['rows']:
   for tid in x['links']:
    t=targets[tid]; shared_a=x['ak']&t['ak'];shared_n=x['nk']&t['nk']; same=x['country']==t['country']; pfx=bool(same and x['prefix5'] and t['nn'][:5]==x['prefix5'])
    a_hit=pfx or any(0<dfa[(x['country'],z)]<=2500 for z in shared_a)
    b_hit=pfx or any(0<dfa[(x['country'],z)]<=25000 for z in shared_a)
    hf_hit=a_hit or (same and any(2500<dfa[(x['country'],z)]<=25000 for z in shared_a) and any(0<dfn[(x['country'],z)]<=25000 for z in shared_n))
    name_hit=a_hit or (same and any(0<dfn[(x['country'],z)]<=2500 for z in shared_n))
    flags={'A':a_hit,'B':b_hit,'A_HF_NAME':hf_hit,'A_NAME2500':name_hit}
    scriptdiff=scripts(x['nn'])!=scripts(t['nn']); empty=not bool(t['address'].strip()); addrhit=any(0<dfa[(x['country'],z)]<=2500 for z in shared_a) if same else False; name_mech=pfx or any(0<dfn[(x['country'],z)]<=2500 for z in shared_n)
    if empty:target_missing+=1
    if x['address'].strip() and t['address'].strip():both_present+=1
    if same and x['address'].strip() and t['address'].strip() and not addrhit:addr_fail+=1
    if empty:
     hfrec_missing+=int(hf_hit);namerec_missing+=int(name_hit)
    if same and x['address'].strip() and t['address'].strip() and not addrhit:
     hfrec_addrfail+=int(hf_hit);namerec_addrfail+=int(name_hit)
    token_equal=set(x['nn'].split())==set(t['nn'].split()); typoish=x['nn']!=t['nn'] and len(x['nn'])>0 and len(t['nn'])>0 and __import__('difflib').SequenceMatcher(None,x['nn'],t['nn']).ratio()>=.72
    for rule,hit in flags.items():
     if hit:continue
     summary[rule]['misses']+=1
     if scriptdiff:summary[rule]['script_difference_misses']+=1
     if rule=='A_HF_NAME': name_allowed=pfx or any(0<dfn[(x['country'],z)]<=25000 for z in shared_n)
     elif rule=='A_NAME2500': name_allowed=pfx or any(0<dfn[(x['country'],z)]<=2500 for z in shared_n)
     else: name_allowed=pfx
     if rule=='A_HF_NAME': address_allowed=any(2500<dfa[(x['country'],z)]<=25000 for z in shared_a)
     else: address_allowed=any(0<dfa[(x['country'],z)]<=(2500 if rule in ('A','A_NAME2500') else 25000) for z in shared_a)
     if not name_allowed:summary[rule]['name_mechanism_failures']+=1
     if not address_allowed:summary[rule]['address_mechanism_failures']+=1
     if not name_allowed and not address_allowed:summary[rule]['both_mechanisms_fail']+=1
     if empty:summary[rule]['target_address_empty_misses']+=1
     if same and x['address'].strip() and t['address'].strip() and not addrhit:summary[rule]['both_addresses_present_address_block_misses']+=1
     c=[]
     if scriptdiff:c.append('multilingual/script')
     if empty:c.append('target address empty')
     if token_equal and x['nn']!=t['nn']:c.append('word order/punctuation')
     if typoish:c.append('typo/abbreviation-like string variation')
     if shared_a and not any(0<dfa[(x['country'],z)]<=2500 for z in shared_a):c.append('shared address keys exceed A cap')
     if not shared_a:c.append('no shared normalized address token')
     if not pfx:c.append('prefix-5 miss')
     if not c:c.append('other')
     cats[rule].update(c)
     if len(examples[rule])<20: examples[rule].append({'s1_id':x['id'],'target_source':t['source'],'target_id':tid,'s1_name':x['name'],'target_name':t['name'],'s1_address':x['address'],'target_address':t['address'],'shared_address_tokens':sorted(shared_a),'shared_name_tokens':sorted(shared_n),'scripts_s1':scripts(x['nn']),'scripts_target':scripts(t['nn']),'script_difference':scriptdiff,'p5_hit':pfx,'address_a_hit':addrhit,'strategy_recovered':False,'annotations':c})
  report.append({'sample':ix,'true_links':sum(len(x['links']) for x in sm['rows']),'overall_address_observations':{'target_address_empty_true_links':target_missing,'both_addresses_present_true_links':both_present,'both_present_address_block_fails_A':addr_fail,'A_HF_NAME_recovers_empty_target_address_links':hfrec_missing,'A_NAME2500_recovers_empty_target_address_links':namerec_missing,'A_HF_NAME_recovers_both_present_address_block_misses':hfrec_addrfail,'A_NAME2500_recovers_both_present_address_block_misses':namerec_addrfail},'strategies':{r:{**summary[r],'overlapping_failure_annotations':dict(cats[r]),'representative_misses':examples[r]} for r in summary}})
 return {'run_tag':tag,'samples':report}
def main():
 a=scan('diagnostic_run1');b=scan('diagnostic_run2');
 (OUT/'diagnostics_run1.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n');(OUT/'diagnostics_run2.json').write_text(json.dumps(b,ensure_ascii=False,indent=2)+'\n')
 strip=lambda x: json.dumps({k:v for k,v in x.items() if k!='run_tag'},sort_keys=True,ensure_ascii=False)
 passed=strip(a)==strip(b);(OUT/'diagnostic_reproducibility.json').write_text(json.dumps({'passed':passed,'exact_diagnostic_reproduction':passed},indent=2)+'\n')
 if not passed:raise RuntimeError('Diagnostic rerun mismatch')
 print('diagnostic_reproducibility_passed',passed)
if __name__=='__main__':main()
