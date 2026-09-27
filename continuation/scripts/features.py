"""Small interpretable pair features, using the approved Stage 3 utilities."""
from __future__ import annotations
from difflib import SequenceMatcher
from normalization.normalize import normalize_name, normalize_address, normalize_address_abbreviations

def _sets(value):
    return set((value or '').split())
def _overlap(a,b):
    if not a or not b:return 0.0
    return len(a&b)/min(len(a),len(b))
def pair_features(s1:dict,target:dict)->dict:
    sn=normalize_name(s1['business_name']) or ''; tn=normalize_name(target['business_name']) or ''
    sa=normalize_address(s1.get('business_address')) or ''; ta=normalize_address(target.get('business_address')) or ''
    saa=normalize_address_abbreviations(s1.get('business_address')) or ''; taa=normalize_address_abbreviations(target.get('business_address')) or ''
    ns,nt=_sets(sn),_sets(tn); ads,adt=_sets(sa),_sets(ta)
    name_exact=bool(sn and sn==tn); addr_exact=bool(sa and ta and sa==ta); addr_abbr_exact=bool(saa and taa and saa==taa)
    return {'name_exact':int(name_exact),'name_char_similarity':SequenceMatcher(None,sn,tn,autojunk=False).ratio() if sn or tn else 0.0,
            'name_token_jaccard':len(ns&nt)/len(ns|nt) if ns|nt else 0.0,'name_token_containment':_overlap(ns,nt),
            'address_exact':int(addr_exact),'address_abbreviation_exact':int(addr_abbr_exact),
            'address_token_jaccard':len(ads&adt)/len(ads|adt) if ads|adt else 0.0,'address_token_containment':_overlap(ads,adt),
            'country_equal':int(s1['country']==target['country']),
            'address_missing_either':int(not (s1.get('business_address') or '').strip() or not (target.get('business_address') or '').strip()),
            'name_exact_agreement':int(name_exact),'address_exact_agreement':int(addr_exact)}
