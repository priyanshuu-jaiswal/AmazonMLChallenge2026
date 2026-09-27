"""Frozen deterministic Stage 5 first-pass weighted scorer."""
WEIGHTS={'name_exact':4.0,'name_char_similarity':3.0,'name_token_jaccard':2.0,'name_token_containment':1.0,
         'address_exact':4.0,'address_abbreviation_exact':3.5,'address_token_jaccard':2.5,'address_token_containment':1.0,
         'country_equal':1.0,'address_missing_either':-0.5}
def score(features): return sum(WEIGHTS[k]*features[k] for k in WEIGHTS)
