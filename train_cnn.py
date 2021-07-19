import json
import torch
import opennre
import sys
import os
import argparse
import logging
import random
import numpy as np

def set_seed(seed):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='', help='Checkpoint name.')
parser.add_argument('--only_test', action='store_true', help='Only run test.')

# data
parser.add_argument('--metric', default='auc', choices=['micro_f1', 'auc'], help='Metric for picking up best checkpoint.')
parser.add_argument('--dataset', default='', choices=['dti', 'biorel', 'GDAb', 'GDAt'], help='Dataset. If not none, the following args can be ignored.')
parser.add_argument('--train_file', default='', type=str, help='Training data file.')
parser.add_argument('--val_file', default='', type=str, help='Validation data file.')
parser.add_argument('--test_file', default='', type=str, help='Test data file.')
parser.add_argument('--rel2id_file', default='', type=str, help='Relation to ID file.')

# bag related
parser.add_argument('--bag_size', type=int, default=3, help='Fixed bag size. If set to 0, use original bag sizes.')
parser.add_argument('--bag_strategy', default='', choices=['ave', 'att'], help='Bag strategy.')

# hyper-parameters
parser.add_argument('--batch_size', default=64, type=int, help='Batch size.')
parser.add_argument('--lr', default=0.2, type=float, help='Learning rate.')
parser.add_argument('--optim', default='sgd', type=str, help='Optimizer.')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay.')
parser.add_argument('--hidden_size', default=250, type=int, help='Hidden layer size.')
parser.add_argument('--max_length', default=100, type=int, help='Maximum sentence length.')
parser.add_argument('--max_epoch', default=20, type=int, help='Max number of training epochs.')

# others
parser.add_argument('--seed', default=42, type=int, help='Random seed.')

args = parser.parse_args()


def main():
    # Set random seed
    set_seed(args.seed)

    # Some basic settings
    root_path = '.'
    sys.path.append(root_path)
    if not os.path.exists('ckpt'):
        os.mkdir('ckpt')
    if len(args.ckpt) == 0:
        args.ckpt = '{}_{}'.format(args.dataset, 'cnn_' + args.bag_strategy)
    ckpt = 'ckpt/{}.pth.tar'.format(args.ckpt)

    if args.dataset != 'none':
        args.train_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_train.txt'.format(args.dataset))
        args.val_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_val.txt'.format(args.dataset))
        if not os.path.exists(args.val_file):
            logging.info("Cannot find the validation file. Use the test file instead.")
            args.val_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_test.txt'.format(args.dataset))
        args.test_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_test.txt'.format(args.dataset))
        args.rel2id_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_rel2id.json'.format(args.dataset))
    else:
        if not (os.path.exists(args.train_file) and os.path.exists(args.val_file) and os.path.exists(args.test_file) and os.path.exists(args.rel2id_file)):
            raise Exception('--train_file, --val_file, --test_file and --rel2id_file are not specified or files do not exist. Or specify --dataset')

    logging.info('Arguments:')
    for arg in vars(args):
        logging.info('    {}: {}'.format(arg, getattr(args, arg)))

    rel2id = json.load(open(args.rel2id_file))

    # get biomedical word2vec embeddings
    if args.dataset == 'dti':  # dti dataset uses different word embeddings than other datasets
        word2id = json.load(open(os.path.join(root_path, 'pretrain/biow2v', 'dti', 'biow2v_word2id.json')))
        word2vec = np.load(os.path.join(root_path, 'pretrain/biow2v', 'dti', 'biow2v_mat.npy'))
    else:
        word2id = json.load(open(os.path.join(root_path, 'pretrain/biowordvec', args.dataset, 'biowordvec_word2id.json')))
        word2vec = np.load(os.path.join(root_path, 'pretrain/biowordvec', args.dataset, 'biowordvec_mat.npy'))

    # Define the sentence encoder
    sentence_encoder = opennre.encoder.CNNEncoder(
        token2id=word2id,
        max_length=args.max_length,
        word_size=200,  
        position_size=10,
        hidden_size=args.hidden_size,
        blank_padding=True,
        kernel_size=3,
        padding_size=1,
        word2vec=word2vec,
        dropout=0.5
    )

    # Define the model
    if args.bag_strategy == 'att':  # use attention strategy to aggregate sentences in bag
        model = opennre.model.BagCNNAttention(sentence_encoder, len(rel2id), rel2id)
    else:  # use average instead
        model = opennre.model.BagCNNAverage(sentence_encoder, len(rel2id), rel2id)

    # Define the whole training framework
    framework = opennre.framework.BagRE(
        train_path=args.train_file,
        val_path=args.val_file,
        test_path=args.test_file,
        model=model,
        ckpt=ckpt,
        batch_size=args.batch_size,
        max_epoch=args.max_epoch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        opt=args.optim,
        bag_size=args.bag_size)

    # Train the model
    if not args.only_test:
        framework.train_model(args.metric)

    # Test the model
    framework.load_state_dict(torch.load(ckpt)['state_dict'])
    result = framework.eval_model(framework.test_loader)

    # Print the result
    logging.info('Test set results:')
    logging.info('AUC: {}'.format(result['auc']))
    logging.info('Micro F1: {}'.format(result['micro_f1']))


if __name__ == '__main__':
    main()
