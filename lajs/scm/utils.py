import json

def read_json(f):
    return json.load(open(f, encoding='utf8'))
def read_jsonl(f):
    return [json.loads(k) for k in open(f, encoding='utf8')]