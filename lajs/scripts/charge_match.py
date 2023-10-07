"""
Extract charges and check whether they match with those of similar cases
"""
# %%
import numpy as np
import json
from pathlib import Path
import re
import pandas as pd
from collections import Counter

import context
from scm.utils import read_json, read_jsonl
from scm.data_utils.read import load_trec, load_lecard_v2
# %%
stage_dir = Path('../datasets/stage_1')
train_ds = read_jsonl(stage_dir / 'train/train_query.json')
test_ds = read_jsonl(stage_dir / 'test/test_query.json')
train_trec = load_trec(stage_dir / 'train/train_label.trec')
cdd_cases = load_lecard_v2('../datasets/lecard_v2')

# %%
all_charges = list(set([c for k in cdd_cases for c in k['charge']]))
print(len(all_charges))
# %%
def get_charge_em(text):
    return [c for c in all_charges if c in text]
# %%
tr_ch = [get_charge_em(d['query']) for d in train_ds]
# %%
part = [k for k in tr_ch if len(k) == 0]
# %%
tr_df = pd.DataFrame(train_ds)
tr_df['charge_em'] = tr_ch
# %%
cdd_df = pd.DataFrame(cdd_cases)
tr_lab = pd.DataFrame(train_trec)
# %%
def get_sim_case_charge(qid):
    part = tr_lab[(tr_lab['qid'] == qid) & (tr_lab['label'] > 0)]
    assert len(part) > 0
    chs = []
    for pid in part['pid'].unique():
        scase = cdd_df[cdd_df['pid'] == pid]
        chs.extend(scase.iloc[0]['charge'])
    return list(set(chs))
tr_df['charge_sim'] = tr_df.apply(lambda k: get_sim_case_charge(k['id']), axis = 1)
# %%
part = tr_df[tr_df['charge_em'].apply(lambda k: len(k) == 0)]

# %%
print(part.iloc[0]['fact'])
# %%
