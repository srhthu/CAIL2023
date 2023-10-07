import json
from multiprocessing import Pool

def read_json(f):
    return json.load(open(f, encoding='utf8'))
def read_jsonl(f):
    return [json.loads(k) for k in open(f, encoding='utf8')]

def mp_map(foo, tasks, n, chunksize = 1):
    pool = Pool(n)
    res = pool.map(foo, tasks, chunksize=chunksize)
    pool.close()
    pool.join()
    return res