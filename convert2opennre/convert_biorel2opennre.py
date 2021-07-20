import json

from os.path import dirname, abspath

parent_dir = dirname(dirname(abspath(__file__)))
with open(parent_dir + '/benchmark/biorel/train.json', 'r') as trf:
    train = json.load(trf)

with open(parent_dir + '/benchmark/biorel/dev.json', 'r') as vdf:
    valid = json.load(vdf)

with open(parent_dir + '/benchmark/biorel/test.json', 'r') as ttf:
    test = json.load(ttf)

with open(parent_dir + '/benchmark/biorel/biorel_train.txt', 'w') as trout:
    for it in train:
        instance = {'text': it['sentence'],
                    'relation': it['relation'],
                    'h': {'id': it['head']['CUI'], 'name': it['head']['word'], 'pos': [it['head']['start'], it['head']['start']+it['head']['length']]},
                    't': {'id': it['tail']['CUI'], 'name': it['tail']['word'], 'pos': [it['tail']['start'], it['tail']['start']+it['tail']['length']]}}
        json.dump(instance, trout)
        trout.write('\n')

with open(parent_dir + '/benchmark/biorel/biorel_val.txt', 'w') as dvout:
    for it in valid:
        instance = {'text': it['sentence'],
                    'relation': it['relation'],
                    'h': {'id': it['head']['CUI'], 'name': it['head']['word'], 'pos': [it['head']['start'], it['head']['start']+it['head']['length']]},
                    't': {'id': it['tail']['CUI'], 'name': it['tail']['word'], 'pos': [it['tail']['start'], it['tail']['start']+it['tail']['length']]}}
        json.dump(instance, dvout)
        dvout.write('\n')

with open(parent_dir + '/benchmark/biorel/biorel_test.txt', 'w') as ttout:
    for it in test:
        instance = {'text': it['sentence'],
                    'relation': it['relation'],
                    'h': {'id': it['head']['CUI'], 'name': it['head']['word'], 'pos': [it['head']['start'], it['head']['start']+it['head']['length']]},
                    't': {'id': it['tail']['CUI'], 'name': it['tail']['word'], 'pos': [it['tail']['start'], it['tail']['start']+it['tail']['length']]}}
        json.dump(instance, ttout)
        ttout.write('\n')

