import json


with open('./benchmark/dti/train.json', 'r') as trf:
    train = json.load(trf)

with open('./benchmark/dti/valid.json', 'r') as vdf:
    valid = json.load(vdf)

with open('./benchmark/dti/test.json', 'r') as ttf:
    test = json.load(ttf)

with open('./benchmark/dti/dti_train.txt', 'w') as trout:
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

with open('./benchmark/dti/dti_val.txt', 'w') as dvout:
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

with open('./benchmark/dti/dti_test.txt', 'w') as ttout:
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

