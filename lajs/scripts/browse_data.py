# %%
import numpy as np
import json
from pathlib import Path
import re
import pandas as pd

import context
from scm.utils import read_json, read_jsonl
from scm.data_utils.read import load_trec

# %%
stage_dir = Path('../datasets/stage_1')
train_ds = read_jsonl(stage_dir / 'train/train_query.json')
train_trec = load_trec(stage_dir / 'train/train_label.trec')
# %%
train_ds[0]
# %%
def fact_in_query(d):
    return d['fact'] in d['query']

part = [d for d in train_ds if not fact_in_query(d)]
# %%
start = re.search('公诉机关指控', part[0]['query']).span()[0]
q_fact = part[0]['query'][start:]
fact = part[0]['fact']
# %%
