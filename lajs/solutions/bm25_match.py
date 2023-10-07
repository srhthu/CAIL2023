"""
Use bm25 to match the text similarity.

Run from the parent dir of this dir.
"""
import numpy as np
import json
from pathlib import Path
import re
import pandas as pd
from collections import Counter
import jieba
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# import context
from scm.utils import read_json, read_jsonl, mp_map
from scm.data_utils.read import load_trec, load_lecard_v2

NUM_MATCH = 30

def tokenize(text):
    return list(jieba.cut(text, cut_all = True))

def main():
    # Load data
    stage_dir = Path('./datasets/stage_1')
    # train_ds = read_jsonl(stage_dir / 'train/train_query.json')
    test_ds = read_jsonl(stage_dir / 'test/test_query.json')
    # train_trec = load_trec(stage_dir / 'train/train_label.trec')
    cdd_cases = load_lecard_v2(stage_dir / 'candidate_55192')

    match_field = 'query' # query or fact

    # Build bm25 corpus
    print('Tokenize corpus...')
    cdd_queries = [k['qw'] for k in cdd_cases][:10000]
    print(len(cdd_queries))

    # corpus = mp_map(tokenize, cdd_queries, 10)
    corpus = []
    for k in tqdm(cdd_queries, ncols=80):
        corpus.append(tokenize(k))
    bm25 = BM25Okapi(corpus)

    results = {}
    for sample in tqdm(test_ds, ncols = 80, disable=True):
        print('    tokenize')
        query = tokenize(sample[match_field])
        print('    get_score')
        scores = bm25.get_scores(query)
        print('    flip')
        rank_ids = np.flip(np.argsort(scores))[:NUM_MATCH]
        pids = [str(cdd_cases[k]['pid']) for k in rank_ids]
        results[str(sample['id'])] = pids
    
    save_dir = Path(f'outputs/stage_1/bm25_{match_field}')
    save_dir.mkdir(parents=True, exist_ok = True)
    with open(save_dir / 'prediction.json', 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()