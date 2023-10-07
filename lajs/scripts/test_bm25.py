"""
Use bm25 to match the text similarity.

Run from the parent dir of this dir.
"""
# %%
import numpy as np
import json
import sys
from pathlib import Path
import re
import pandas as pd
from collections import Counter
import jieba
from rank_bm25 import BM25Okapi
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

import context
from scm.utils import read_json, read_jsonl, mp_map
from scm.data_utils.read import load_trec, load_lecard_v2
# %%
NUM_MATCH = 30
match_field = 'query' # query or fact

def tokenize(text):
    return list(jieba.cut(text, cut_all = True))
# %%

# Load data
stage_dir = Path('../datasets/stage_1')
test_ds = read_jsonl(stage_dir / 'test/test_query.json')
cdd_cases = load_lecard_v2(stage_dir / 'candidate_55192')
# %%
corpus = [tokenize(k['qw']) for k in cdd_cases]
# %%
# Build bm25 corpus
print('Tokenize corpus...')
cdd_queries = [k['qw'] for k in cdd_cases]
print(len(cdd_queries))

corpus = mp_map(tokenize, cdd_queries, 10)
# %%
bm25 = BM25Okapi(corpus)
# %%
dct = Dictionary(corpus)  # fit dictionary
corpus = [dct.doc2bow(line) for line in corpus]
model = TfidfModel(corpus)
corpus_vec = model[corpus]
# %%
def get_sp_len(x):
    # get the length of a sparse vector
    s = np.square([k[1] for k in x]).sum()
    return np.sqrt(s)
corpus_vec_norm = [get_sp_len(k) for k in corpus_vec]
# %%
def cosine_score(spv):
    id2v = {k[0]:k[1] for k in spv}
    q_len = get_sp_len(spv)
    scores = []
    for c_vec, c_vlen in zip(corpus_vec, corpus_vec_norm):
        c_id2v = {k[0]:k[1] for k in c_vec}
        common_k = set(id2v.keys()).intersection(set(c_id2v.keys()))
        q_v = np.array([id2v[i] for i in common_k])
        c_v = np.array([c_id2v[i] for i in common_k])
        sc = np.sum(q_v * c_v)
        scores.append(sc / q_len / c_vlen)
    return np.array(scores)
# %%
def search_bm25(example):
    tk_query = tokenize(example[match_field])
    scores = bm25.get_scores(tk_query)
    rank_ids = np.flip(np.argsort(scores))[:NUM_MATCH]
    pids = [str(cdd_cases[k]['pid']) for k in rank_ids]
    return (str(example['id']), pids)
def search_tfidf(example):
    query = tokenize(example[match_field])
    query = dct.doc2bow(query)
    scores = cosine_score(model[query])
    rank_ids = np.flip(np.argsort(scores))[:NUM_MATCH]
    pids = [str(cdd_cases[k]['pid']) for k in rank_ids]
    return (str(example['id']), pids)
# %%
start = time.time()
search_tfidf(test_ds[0])
dur = time.time() - start
print(dur)
# %%
results = {}
for sample in tqdm(test_ds, ncols = 80, disable=False):
    query = tokenize(sample[match_field])
    query = dct.doc2bow(query)
    scores = cosine_score(model[query])
    # scores = bm25.get_scores(query)
    
    rank_ids = np.flip(np.argsort(scores))[:NUM_MATCH]
    pids = [str(cdd_cases[k]['pid']) for k in rank_ids]
    results[str(sample['id'])] = pids

save_dir = Path(f'outputs/stage_1/bm25_{match_field}')
save_dir.mkdir(parents=True, exist_ok = True)
# %%
with open(save_dir / 'prediction.json', 'w') as f:
    json.dump(results, f)
