
from collections import Counter
import numpy as np
import logging


def read_conll(in_file, lowercase=True, max_example=None):
    """
        Load parse trees from CoNLL file.
        See CoNLL-X format at http://ilk.uvt.nl/conll/.
    """
    examples = []
    with open(in_file) as f:
        word, pos, head, label = [], [], [], []
        for line in f.readlines():
            sp = line.strip().split('\t')
            if len(sp) == 10:
                word.append(sp[1].lower() if lowercase else sp[1])
                pos.append(sp[4])
                head.append(int(sp[6]))
                label.append(sp[7])
            elif len(word) > 0:
                examples.append({'word': word, 'pos': pos, 'head': head, 'label': label})
                word, pos, head, label = [], [], [], []
                if (max_example is not None) and (len(examples) == max_example):
                    break
        if len(word) > 0:
            examples.append({'word': word, 'pos': pos, 'head': head, 'label': label})
    logging.info('#examples: %d' % len(examples))
    return examples


def build_dict(keys, n_max=None, offset=0):
    """
        Build a dictionary of a list of keys.
    """
    count = Counter()
    for key in keys:
        count[key] += 1
    ls = count.most_common() if n_max is None \
        else count.most_common(n_max)

    logging.info('build_dict: %d keys, kept the most common %d ones.' % (len(count), len(ls)))
    for key in ls[:5]:
        logging.info(key)
    logging.info('...')
    for key in ls[-5:]:
        logging.info(key)

    # leave 0 to UNK
    return {w[0]: index + offset for (index, w) in enumerate(ls)}


def get_embeddings(in_file):
    """
        Load embedding file.
    """
    embeddings = {}
    for line in open(in_file).readlines():
        sp = line.strip().split()
        embeddings[sp[0]] = [float(x) for x in sp[1:]]
    return embeddings


def get_minibatches(n, minibatch_size, shuffle=True):
    idx_list = np.arange(0, n, minibatch_size)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for idx in idx_list:
        minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
    return minibatches
