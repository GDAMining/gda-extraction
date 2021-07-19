import json


with open('./data/biorel/train.json', 'r') as trf:
    train = json.load(trf)

with open('./data/biorel/dev.json', 'r') as vdf:
    valid = json.load(vdf)

with open('./data/biorel/test.json', 'r') as ttf:
    test = json.load(ttf)

with open('./benchmark/biorel/biorel_train.txt', 'w') as trout:
    for it in train:
        instance = {'text': it['sentence'],
                    'relation': it['relation'],
                    'h': {'id': it['head']['CUI'], 'name': it['head']['word'], 'pos': [it['head']['start'], it['head']['start']+it['head']['length']]},
                    't': {'id': it['tail']['CUI'], 'name': it['tail']['word'], 'pos': [it['tail']['start'], it['tail']['start']+it['tail']['length']]}}
        json.dump(instance, trout)
        trout.write('\n')

with open('./benchmark/biorel/biorel_val.txt', 'w') as dvout:
    for it in valid:
        instance = {'text': it['sentence'],
                    'relation': it['relation'],
                    'h': {'id': it['head']['CUI'], 'name': it['head']['word'], 'pos': [it['head']['start'], it['head']['start']+it['head']['length']]},
                    't': {'id': it['tail']['CUI'], 'name': it['tail']['word'], 'pos': [it['tail']['start'], it['tail']['start']+it['tail']['length']]}}
        json.dump(instance, dvout)
        dvout.write('\n')

with open('./benchmark/biorel/biorel_test.txt', 'w') as ttout:
    for it in test:
        instance = {'text': it['sentence'],
                    'relation': it['relation'],
                    'h': {'id': it['head']['CUI'], 'name': it['head']['word'], 'pos': [it['head']['start'], it['head']['start']+it['head']['length']]},
                    't': {'id': it['tail']['CUI'], 'name': it['tail']['word'], 'pos': [it['tail']['start'], it['tail']['start']+it['tail']['length']]}}
        json.dump(instance, ttout)
        ttout.write('\n')

