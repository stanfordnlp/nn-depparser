import argparse
import os
import theano
from sys import platform as _platform


DATA_DIR = '/Users/danqi/Documents/research/datasets/dependency-treebanks/' \
    if _platform == 'darwin' \
    else '/u/nlp/data/dependency_treebanks/'

LOG_DIR = '../logs/'
MODEL_DIR = '../models/'

_floatX = theano.config.floatX

P_PREFIX = '<p>:'
L_PREFIX = '<l>:'
UNK = '<UNK>'
NULL = '<NULL>'
ROOT = '<ROOT>'


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_args():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    parser.add_argument('--job_id', '-j',
                        type=str,
                        default=None,
                        help='The job id to run')

    parser.add_argument('--random_seed',
                        type=int,
                        default=1013,
                        help='Random seed')

    parser.add_argument('--data_path',
                        type=str,
                        default=os.path.join(DATA_DIR, 'PTB/Stanford_3_3_0'),
                        help='Data path')

    parser.add_argument('--train_file',
                        type=str,
                        default='train.conll',
                        help='Training file')

    parser.add_argument('--dev_file',
                        type=str,
                        default='dev.conll',
                        help='Dev file')

    parser.add_argument('--embedding_file',
                        type=str,
                        default=os.path.join(DATA_DIR, 'embeddings/en-cw.txt'))

    parser.add_argument('--lowercase',
                        type='bool',
                        default=False,
                        help='Whether to make words lowercase by default')

    parser.add_argument('--max_train',
                        type=int,
                        default=None,
                        help='Only use the first max_train examples for training, default is None')

    parser.add_argument('--hidden_size',
                        type=int,
                        default=200,
                        help='Hidden size')

    parser.add_argument('--embedding_size',
                        type=int,
                        default=50,
                        help='Embedding size')

    parser.add_argument('--batch_size',
                        type=int,
                        default=100,
                        help='Batch size')

    parser.add_argument('--n_epoches',
                        type=int,
                        default=100,
                        help='Number of epoches')

    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.1)

    #TODO: add l2 regularization

    parser.add_argument('--eval_iter',
                        type=int,
                        default=100,
                        help='Evaluation on dev set after K updates')

    parser.add_argument('--dropout_rate',
                        type=float,
                        default=0.5,
                        help='Dropout rate')

    parser.add_argument('--optimizer',
                        type=str,
                        default='adam',
                        help='Optimizer')

    parser.add_argument('--max_words',
                        type=int,
                        default=None,
                        help='Keep the most frequent max_words word types, default is None')

    parser.add_argument('--unlabeled',
                        type='bool',
                        default=True,
                        help='Whether to train an unlabeled parser, default is True')

    parser.add_argument('--use_pos',
                        type='bool',
                        default=True,
                        help='Whether to use the part-of-speech tags, defauls is True')

    parser.add_argument('--use_dep',
                        type='bool',
                        default=False,
                        help='Whether to use the dependency labels, defauls is False')

    #TODO: havent used single_root yet..
    parser.add_argument('--single_root',
                        type='bool',
                        default=True,
                        help='Whether to allow multiple roots.')

    return parser.parse_args()
