import json

from os.path import dirname, abspath

parent_dir = dirname(dirname(abspath(__file__)))
with open(parent_dir + '/benchmark/dti/train.json', 'r') as trf:
    train = json.load(trf)

with open(parent_dir + '/benchmark/dti/valid.json', 'r') as vdf:
    valid = json.load(vdf)

with open(parent_dir + '/benchmark/dti/test.json', 'r') as ttf:
    test = json.load(ttf)

with open(parent_dir + '/benchmark/dti/dti_train.txt', 'w') as trout:
    for it in train:
        head_start = it['sentence'].find(it['head']['word'])
        assert head_start != -1
        tail_start = it['sentence'].find(it['tail']['word'])
        assert tail_start != -1
        instance = {'text': it['sentence'],
                    'relation': it['relation'],
                    'h': {'id': it['head']['id'], 'name': it['head']['word'], 'pos': [head_start, head_start+len(it['head']['word'])]},
                    't': {'id': it['tail']['id'], 'name': it['tail']['word'], 'pos': [tail_start, tail_start+len(it['tail']['word'])]}}
        json.dump(instance, trout)
        trout.write('\n')

with open(parent_dir + '/benchmark/dti/dti_val.txt', 'w') as dvout:
    for it in valid:
        head_start = it['sentence'].find(it['head']['word'])
        assert head_start != -1
        tail_start = it['sentence'].find(it['tail']['word'])
        assert tail_start != -1
        instance = {'text': it['sentence'],
                    'relation': it['relation'],
                    'h': {'id': it['head']['id'], 'name': it['head']['word'], 'pos': [head_start, head_start+len(it['head']['word'])]},
                    't': {'id': it['tail']['id'], 'name': it['tail']['word'], 'pos': [tail_start, tail_start+len(it['tail']['word'])]}}
        json.dump(instance, dvout)
        dvout.write('\n')

with open(parent_dir + '/benchmark/dti/dti_test.txt', 'w') as ttout:
    for it in test:
        head_start = it['sentence'].find(it['head']['word'])
        assert head_start != -1
        tail_start = it['sentence'].find(it['tail']['word'])
        assert tail_start != -1
        instance = {'text': it['sentence'],
                    'relation': it['relation'],
                    'h': {'id': it['head']['id'], 'name': it['head']['word'], 'pos': [head_start, head_start+len(it['head']['word'])]},
                    't': {'id': it['tail']['id'], 'name': it['tail']['word'], 'pos': [tail_start, tail_start+len(it['tail']['word'])]}}
        json.dump(instance, ttout)
        ttout.write('\n')

with open(parent_dir + '/benchmark/dti/dti_rel2id.json', 'w') as rel2out:
    json.dump({"NA": 0, "substrate": 1, "inhibitor": 2, "agonist": 3, "unknown": 4, "other": 5}, rel2out)
