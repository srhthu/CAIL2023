from scm.utils import read_json, read_jsonl

def load_trec(path):
    def parse(line:str):
        eles = line.strip().split('\t')
        return {'qid': eles[0], 'pid': eles[2], 'label': eles[3]}
    with open(path, encoding='utf8') as f:
        data = list(map(parse, iter(f)))
    return data