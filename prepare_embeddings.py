import os
import json
import argparse
import jsonlines
import numpy as np

from tqdm import tqdm
from opennre.tokenization import utils
from gensim.models import KeyedVectors

flags = argparse.ArgumentParser()
flags.add_argument("--embs_fpath", default='./pretrain/insert-embeddings-folder-here/',
                   type=str, help="embeddings folder path.")
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

    text = text.lower()
    text = utils.convert_to_unicode(text)
    text = utils.clean_text(text)
    text = utils.tokenize_chinese_chars(text)
    tknz_text = utils.split_on_whitespace(text)
    return tknz_text


def main():
    # get benchmark name
    benchmark = FLAGS.benchmark_fpath.split('/')[-2]
    # get embeddings filename 
    embs_fname = FLAGS.embs_fpath.split('/')[-2] + '.bin'
    # read training, validation and test sets
    print('Reading training, validation, and test sets...')
    dataset = []
    with jsonlines.open(FLAGS.benchmark_fpath + benchmark + '_train.txt', 'r') as train_reader:  # training instances
        for row in train_reader:
            dataset.append(row)

    with jsonlines.open(FLAGS.benchmark_fpath + benchmark + '_val.txt', 'r') as val_reader:  # validation instances
        for row in val_reader:
            dataset.append(row)

    with jsonlines.open(FLAGS.benchmark_fpath + benchmark + '_test.txt', 'r') as test_reader:  # test instances
        for row in test_reader:
            dataset.append(row)
    print('Training, validation, and test sets read!')

    # get word2vec model stored in C *binary* format
    print('Loading word2vec...')
    word2vec = KeyedVectors.load_word2vec_format(FLAGS.embs_fpath + embs_fname, binary=True)
    print('Word2vec loaded!')

    # restrict word embeddings to words contained within dataset
    print('Restricting word embeddings to dataset...')
    word2ix = {}
    vectors = []
    for data in tqdm(dataset):
        # tokenize text data
        tknz_text = tokenize(data["text"])
        for token in tknz_text:
            if token in word2vec and token not in word2ix:  # token found within word2vec dictionary -- keep it
                word2ix[token] = len(word2ix)
                vectors.append(word2vec[token])
    # convert vectors to numpy 2d-array
    vectors = np.array(vectors)
    print('Word embeddings restricted to dataset!')
    print('Restricted to {} vectors'.format(len(word2ix)))
    print('Word embeddings matrix has size {}x{}'.format(vectors.shape[0], vectors.shape[1]))

    # create benchmark-dependent dir if not exists
    if not os.path.exists(FLAGS.embs_fpath + benchmark + '/'):
        os.makedirs(FLAGS.embs_fpath + benchmark + '/')
    # get embeddings name
    embs_name = embs_fname.split('.')[0]
    # store word to index dict and pre-trained vectors
    with open(FLAGS.embs_fpath + benchmark + '/' + embs_name + '_word2id.json', 'w') as dout:
        json.dump(word2ix, dout)
    np.save(FLAGS.embs_fpath + benchmark + '/' + embs_name + '_mat.npy', vectors)


if __name__ == '__main__':
    main()
