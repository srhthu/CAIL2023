from pathlib import Path

from scm.utils import read_json, read_jsonl

def load_trec(path):
    def parse(line:str):
        eles = line.strip().split('\t')
        return {'qid': int(eles[0]), 'pid': int(eles[2]), 'label': int(eles[3])}
    with open(path, encoding='utf8') as f:
        data = list(map(parse, iter(f)))
    return data

def load_lecard_v2(cdd_dir):
    """
    The directory of candidates. `candidate_55192` for stage_1
    """
    file_names = Path(cdd_dir).iterdir()
    data = [read_json(k) for k in file_names]
    return data