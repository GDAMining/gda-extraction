import argparse
import jsonlines

from opennre.tokenization import utils
from statistics import mean, stdev

flags = argparse.ArgumentParser()
flags.add_argument("--benchmark_fpath", default='./benchmark/insert-benchmark-name-here/',
                   type=str, help="benchmark folder path.")
FLAGS = flags.parse_args()


def tokenize(text):
    """
    Tokenize a piece of text as in opennre framework -- not used by BERT-based networks

    Args:
        text (str): a piece of text to tokenize

    Return:
        returns tokenized text
    """

    text = utils.convert_to_unicode(text)
    text = utils.clean_text(text)
    text = utils.tokenize_chinese_chars(text)
    tknz_text = utils.split_on_whitespace(text)
    return tknz_text


def main():
    # get benchmark name
    benchmark_name = FLAGS.benchmark_fpath.split('/')[-2]
    # read training, validation and test sets
    train = []
    valid = []
    test = []
    print('Reading training, validation, and test sets...')
    with jsonlines.open(FLAGS.benchmark_fpath + '/' + benchmark_name + '_train.txt', 'r') as train_reader:  # training instances
        for row in train_reader:
            train.append(row)

    with jsonlines.open(FLAGS.benchmark_fpath + '/' + benchmark_name + '_val.txt', 'r') as val_reader:  # validation instances
        for row in val_reader:
            valid.append(row)

    with jsonlines.open(FLAGS.benchmark_fpath + '/' + benchmark_name + '_test.txt', 'r') as test_reader:  # test instances
        for row in test_reader:
            test.append(row)
    print('Training, validation, and test sets read!')

    # compute statistics over training, validation and test sets
    print('Compute statistics for training, development, and test sets')
    tr_rcount = 0
    tr_bcount = {}
    tr_pcount = {}
    tr_pbcount = {}
    tr_len = []

    dv_rcount = 0
    dv_bcount = {}
    dv_pcount = {}
    dv_pbcount = {}
    dv_len = []

    tt_rcount = 0
    tt_bcount = {}
    tt_pcount = {}
    tt_pbcount = {}
    tt_len = []

    for train_data in train:
        tr_rcount += 1
        tknz_train_data = tokenize(train_data['text'])
        tr_len.append(len(tknz_train_data))
        if train_data['relation'] in tr_pbcount:
            if (train_data['h']['id'], train_data['t']['id']) not in tr_pbcount[train_data['relation']]:
                tr_pbcount[train_data['relation']][train_data['h']['id'], train_data['t']['id']] = 1
        else:
            tr_pbcount[train_data['relation']] = {(train_data['h']['id'], train_data['t']['id']): 1}
        if (train_data['h']['id'], train_data['t']['id']) in tr_bcount:
            tr_bcount[train_data['h']['id'], train_data['t']['id']] += 1
        else:
            tr_bcount[train_data['h']['id'], train_data['t']['id']] = 1
        if train_data['relation'] in tr_pcount:
            tr_pcount[train_data['relation']] += 1
        else:
            tr_pcount[train_data['relation']] = 1
    # compute number of bags per relation
    tr_pbcount = {pred: len(pairs) for pred, pairs in tr_pbcount.items()}
    # compute average number of sentences per bag in training set
    tr_bavg = sum(list(tr_bcount.values())) / len(tr_bcount)
    # compute average length and variance of sentences in training set
    tr_lavg = mean(tr_len)
    tr_lstd = stdev(tr_len)
    # get maximum length of sentences in training set
    tr_lmax = max(tr_len)

    for val_data in valid:
        dv_rcount += 1
        tknz_val_data = tokenize(val_data['text'])
        dv_len.append(len(tknz_val_data))
        if val_data['relation'] in dv_pbcount:
            if (val_data['h']['id'], val_data['t']['id']) not in dv_pbcount[val_data['relation']]:
                dv_pbcount[val_data['relation']][val_data['h']['id'], val_data['t']['id']] = 1
        else:
            dv_pbcount[val_data['relation']] = {(val_data['h']['id'], val_data['t']['id']): 1}
        if (val_data['h']['id'], val_data['t']['id']) in dv_bcount:
            dv_bcount[val_data['h']['id'], val_data['t']['id']] += 1
        else:
            dv_bcount[val_data['h']['id'], val_data['t']['id']] = 1
        if val_data['relation'] in dv_pcount:
            dv_pcount[val_data['relation']] += 1
        else:
            dv_pcount[val_data['relation']] = 1
    # compute number of bags per relation
    dv_pbcount = {pred: len(pairs) for pred, pairs in dv_pbcount.items()}
    # compute average number of sentences per bag in validation set
    dv_bavg = sum(list(dv_bcount.values())) / len(dv_bcount)
    # compute average length and variance of sentences in validation set
    dv_lavg = mean(dv_len)
    dv_lstd = stdev(dv_len)
    # get maximum length of sentences in validation set
    dv_lmax = max(dv_len)

    for test_data in test:
        tt_rcount += 1
        tknz_test_data = tokenize(test_data['text'])
        tt_len.append(len(tknz_test_data))
        if test_data['relation'] in tt_pbcount:
            if (test_data['h']['id'], test_data['t']['id']) not in tt_pbcount[test_data['relation']]:
                tt_pbcount[test_data['relation']][test_data['h']['id'], test_data['t']['id']] = 1
        else:
            tt_pbcount[test_data['relation']] = {(test_data['h']['id'], test_data['t']['id']): 1}
        if (test_data['h']['id'], test_data['t']['id']) in tt_bcount:
            tt_bcount[test_data['h']['id'], test_data['t']['id']] += 1
        else:
            tt_bcount[test_data['h']['id'], test_data['t']['id']] = 1
        if test_data['relation'] in tt_pcount:
            tt_pcount[test_data['relation']] += 1
        else:
            tt_pcount[test_data['relation']] = 1
    # compute number of bags per relation
    tt_pbcount = {pred: len(pairs) for pred, pairs in tt_pbcount.items()}
    # compute average number of sentences per bag in test set
    tt_bavg = sum(list(tt_bcount.values())) / len(tt_bcount)
    # compute average length and variance of sentences in test set
    tt_lavg = mean(tt_len)
    tt_lstd = stdev(tt_len)
    # get maximum length of sentences in test set
    tt_lmax = max(tt_len)

    print('Number of sentences per set:\nTrain: {}\nValidation: {}\nTest: {}'.format(tr_rcount, dv_rcount, tt_rcount))
    print('Average length of sentences per set:\nTrain: {}\nValidation: {}\nTest:{}'.format(tr_lavg, dv_lavg, tt_lavg))
    print('Standard deviation length of sentences per set:\nTrain: {}\nValidation: {}\nTest:{}'.format(tr_lstd, dv_lstd, tt_lstd))
    print('Maximum length of sentences per set:\nTrain: {}\nValidation: {}\nTest:{}\n'.format(tr_lmax, dv_lmax, tt_lmax))
    print('Number of bags per set:\nTrain: {}\nValidation: {}\nTest: {}\n'.format(len(tr_bcount), len(dv_bcount), len(tt_bcount)))
    print('Average number of sentences per bag:\nTrain: {}\nValidation: {}\nTest: {}\n'.format(tr_bavg, dv_bavg, tt_bavg))
    print('Number of relation occurrences per set:\nTrain: {}\nValidation: {}\nTest:{}'.format(tr_pcount, dv_pcount, tt_pcount))
    print('Number of bags per relation:\nTrain: {}\nValidation: {}\nTest:{}'.format(tr_pbcount, dv_pbcount, tt_pbcount))


if __name__ == '__main__':
    main()

