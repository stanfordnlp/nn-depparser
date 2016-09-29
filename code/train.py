#!/usr/bin/python

import os
import sys
import utils
import logging
import time

import config
from config import L_PREFIX, P_PREFIX, UNK, NULL, ROOT
from config import _floatX

import theano
import theano.tensor as T
import lasagne
import numpy as np
from collections import Counter

class CubicLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return input * input * input


class Parser:

    def __init__(self, dataset, args):
        logging.info('Build dictionary for dependency deprel.')
        root_labels = list([l for ex in dataset
                           for (h, l) in zip(ex['head'], ex['label']) if h == 0])
        counter = Counter(root_labels)
        if len(counter) > 1:
            logging.info('Warning: more than one root label')
            logging.info(counter)
        self.root_label = counter.most_common()[0][0]
        deprel = [self.root_label] + list(set([w for ex in dataset
                                               for w in ex['label']
                                               if w != self.root_label]))
        tok2id = {L_PREFIX + l: i for (i, l) in enumerate(deprel)}
        tok2id[L_PREFIX + NULL] = self.L_NULL = len(tok2id)
        logging.info('Root label: %s' % self.root_label)
        logging.info('Labels (%d): %s' % (len(deprel), ', '.join(deprel)))

        self.unlabeled = args.unlabeled
        self.with_punct = args.with_punct
        self.use_pos = args.use_pos
        self.use_dep = args.use_dep
        self.n_layers = args.n_layers
        self.nonlinearity = args.nonlinearity
        self.optimizer = args.optimizer
        self.learning_rate = args.learning_rate
        self.dropout_rate = args.dropout_rate
        self.input_dropout_rate = args.input_dropout_rate
        self.b_init = args.b_init
        self.l2_reg = args.l2_reg
        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.language = args.language

        if self.unlabeled:
            trans = ['L', 'R', 'S']
            self.n_deprel = 1
        else:
            trans = ['L-' + l for l in deprel] + ['R-' + l for l in deprel] + ['S']
            self.n_deprel = len(deprel)

        self.n_trans = len(trans)
        self.tran2id = {t: i for (i, t) in enumerate(trans)}
        self.id2tran = {i: t for (i, t) in enumerate(trans)}

        logging.info('Build dictionary for part-of-speech tags.')
        tok2id.update(utils.build_dict([P_PREFIX + w for ex in dataset for w in ex['pos']],
                                       offset=len(tok2id)))
        tok2id[P_PREFIX + UNK] = self.P_UNK = len(tok2id)
        tok2id[P_PREFIX + NULL] = self.P_NULL = len(tok2id)
        tok2id[P_PREFIX + ROOT] = self.P_ROOT = len(tok2id)

        logging.info('Build dictionary for words.')
        tok2id.update(utils.build_dict([w for ex in dataset for w in ex['word']],
                                       offset=len(tok2id)))
        tok2id[UNK] = self.UNK = len(tok2id)
        tok2id[NULL] = self.NULL = len(tok2id)
        tok2id[ROOT] = self.ROOT = len(tok2id)

        self.tok2id = tok2id
        self.id2tok = {v: k for (k, v) in tok2id.items()}

        self.n_features = 18 + (18 if args.use_pos else 0) + (12 if args.use_dep else 0)
        self.n_tokens = len(tok2id)

        logging.info('#deprels: %d' % self.n_deprel)
        logging.info('#transitions: %d' % self.n_trans)
        logging.info('#features: %d' % self.n_features)
        logging.info('#tokens: %d' % self.n_tokens)

    def vectorize(self, examples):
        """
            Vectorize the examples.
            tok2id includes words, part-of-speech tags and deprel.
            Also add a ROOT token to the front of the sequence.
        """
        vec_examples = []
        for ex in examples:
            word = [self.ROOT] + [self.tok2id[w] if w in self.tok2id
                                  else self.UNK for w in ex['word']]
            pos = [self.P_ROOT] + [self.tok2id[P_PREFIX + w] if P_PREFIX + w in self.tok2id
                                   else self.P_UNK for w in ex['pos']]
            head = [-1] + ex['head']
            label = [-1] + [self.tok2id[L_PREFIX + w] if L_PREFIX + w in self.tok2id
                            else -1 for w in ex['label']]
            vec_examples.append({'word': word, 'pos': pos,
                                 'head': head, 'label': label})
        return vec_examples

    def extract_features(self, stack, buf, arcs, ex):
        """
            Extract features based on stack / buf / arcs.
            Hardcoded for now.
        """
        def get_lc(k):
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] < k])

        def get_rc(k):
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] > k],
                          reverse=True)

        p_features = []
        l_features = []
        features = [self.NULL] * (3 - len(stack)) + [ex['word'][x] for x in stack[-3:]]
        features += [ex['word'][x] for x in buf[:3]] + [self.NULL] * (3 - len(buf))
        if self.use_pos:
            p_features = [self.P_NULL] * (3 - len(stack)) + [ex['pos'][x] for x in stack[-3:]]
            p_features += [ex['pos'][x] for x in buf[:3]] + [self.P_NULL] * (3 - len(buf))

        for i in xrange(2):
            if i < len(stack):
                k = stack[-i-1]
                lc = get_lc(k)
                rc = get_rc(k)
                llc = get_lc(lc[0]) if len(lc) > 0 else []
                rrc = get_rc(rc[0]) if len(rc) > 0 else []

                features.append(ex['word'][lc[0]] if len(lc) > 0 else self.NULL)
                features.append(ex['word'][rc[0]] if len(rc) > 0 else self.NULL)
                features.append(ex['word'][lc[1]] if len(lc) > 1 else self.NULL)
                features.append(ex['word'][rc[1]] if len(rc) > 1 else self.NULL)
                features.append(ex['word'][llc[0]] if len(llc) > 0 else self.NULL)
                features.append(ex['word'][rrc[0]] if len(rrc) > 0 else self.NULL)

                if self.use_pos:
                    p_features.append(ex['pos'][lc[0]] if len(lc) > 0 else self.P_NULL)
                    p_features.append(ex['pos'][rc[0]] if len(rc) > 0 else self.P_NULL)
                    p_features.append(ex['pos'][lc[1]] if len(lc) > 1 else self.P_NULL)
                    p_features.append(ex['pos'][rc[1]] if len(rc) > 1 else self.P_NULL)
                    p_features.append(ex['pos'][llc[0]] if len(llc) > 0 else self.P_NULL)
                    p_features.append(ex['pos'][rrc[0]] if len(rrc) > 0 else self.P_NULL)

                if self.use_dep:
                    l_features.append(ex['label'][lc[0]] if len(lc) > 0 else self.L_NULL)
                    l_features.append(ex['label'][rc[0]] if len(rc) > 0 else self.L_NULL)
                    l_features.append(ex['label'][lc[1]] if len(lc) > 1 else self.L_NULL)
                    l_features.append(ex['label'][rc[1]] if len(rc) > 1 else self.L_NULL)
                    l_features.append(ex['label'][llc[0]] if len(llc) > 0 else self.L_NULL)
                    l_features.append(ex['label'][rrc[0]] if len(rrc) > 0 else self.L_NULL)
            else:
                features += [self.NULL] * 6
                if self.use_pos:
                    p_features += [self.P_NULL] * 6
                if self.use_dep:
                    l_features += [self.L_NULL] * 6

        features += p_features + l_features
        assert len(features) == self.n_features
        return features

    def get_oracle(self, stack, buf, ex):
        if len(stack) < 2:
            return self.n_trans - 1

        i0 = stack[-1]
        i1 = stack[-2]
        h0 = ex['head'][i0]
        h1 = ex['head'][i1]
        l0 = ex['label'][i0]
        l1 = ex['label'][i1]

        if self.unlabeled:
            if (i1 > 0) and (h1 == i0):
                return 0
            elif (i1 >= 0) and (h0 == i1) and \
                 (not any([x for x in buf if ex['head'][x] == i0])):
                return 1
            else:
                return None if len(buf) == 0 else 2
        else:
            if (i1 > 0) and (h1 == i0):
                return l1 if (l1 >= 0) and (l1 < self.n_deprel) else None
            elif (i1 >= 0) and (h0 == i1) and \
                 (not any([x for x in buf if ex['head'][x] == i0])):
                return l0 + self.n_deprel if (l0 >= 0) and (l0 < self.n_deprel) else None
            else:
                return None if len(buf) == 0 else self.n_trans - 1

    def create_instances(self, examples):
        all_instances = []
        succ = 0
        for id, ex in enumerate(examples):
            if id % 1000 == 0:
                logging.info('processed %d examples.' % id)
            n_words = len(ex['word']) - 1

            # arcs = {(h, t, label)}
            stack = [0]
            buf = [i + 1 for i in xrange(n_words)]
            arcs = []
            instances = []
            for i in xrange(n_words * 2):
                gold_t = self.get_oracle(stack, buf, ex)
                if gold_t is None:
                    break
                legal_labels = self.legal_labels(stack, buf)
                assert legal_labels[gold_t] == 1
                instances.append((self.extract_features(stack, buf, arcs, ex),
                                  legal_labels, gold_t))
                if gold_t == self.n_trans - 1:
                    stack.append(buf[0])
                    buf = buf[1:]
                elif gold_t < self.n_deprel:
                    arcs.append((stack[-1], stack[-2], gold_t))
                    stack = stack[:-2] + [stack[-1]]
                else:
                    arcs.append((stack[-2], stack[-1], gold_t - self.n_deprel))
                    stack = stack[:-1]
            else:
                succ += 1
                all_instances += instances

        logging.info('success: %d (%.2f %%)' % (succ, succ * 100.0 / len(examples)))
        logging.info('#Instances: %d' % len(all_instances))
        return all_instances

    def build_fn(self, emb={}, pre_trained_params=None):
        in_x = T.imatrix('x')
        in_y = T.ivector('y')
        in_l = T.matrix('l')

        l_in = lasagne.layers.InputLayer((None, self.n_features), in_x)
        embeddings = np.random.normal(0, 0.01, (self.n_tokens, self.embedding_size)).astype(_floatX)
        n_pre_trained = 0
        for token in self.tok2id:
            i = self.tok2id[token]
            if token in emb:
                embeddings[i] = emb[token]
                n_pre_trained += 1
            elif token.lower() in emb:
                embeddings[i] = emb[token.lower()]
                n_pre_trained += 1
        logging.info('pre-trained: %d / %d = %.2f%%' % (n_pre_trained, self.n_tokens,
                                                        n_pre_trained * 100.0 / self.n_tokens))
        l_emb = lasagne.layers.EmbeddingLayer(l_in, self.n_tokens,
                                              self.embedding_size, W=embeddings)
        network = l_emb
        if self.input_dropout_rate > 0:
            network = lasagne.layers.DropoutLayer(network, p=self.input_dropout_rate)

        for _ in xrange(self.n_layers):
            if self.nonlinearity == 'relu':
                network = lasagne.layers.DenseLayer(network, self.hidden_size,
                                                    b=lasagne.init.Constant(self.b_init),
                                                    nonlinearity=lasagne.nonlinearities.rectify)
            elif self.nonlinearity == 'tanh':
                network = lasagne.layers.DenseLayer(network, self.hidden_size,
                                                    nonlinearity=lasagne.nonlinearities.tanh)
            elif self.nonlinearity == 'cubic':
                network = lasagne.layers.DenseLayer(network, self.hidden_size,
                                                    nonlinearity=None)
                network = CubicLayer(network)
            else:
                raise NotImplementedError('nonlinearity = %s' % self.nonlinearity)
            if self.dropout_rate > 0:
                network = lasagne.layers.DropoutLayer(network, p=self.dropout_rate)

        network = lasagne.layers.DenseLayer(network, self.n_trans, b=None,
                                            nonlinearity=lasagne.nonlinearities.softmax)

        if pre_trained_params is not None:
            lasagne.layers.set_all_param_values(network, pre_trained_params, trainable=True)

        # train_prob = lasagne.layers.get_output(network, deterministic=False) * in_l
        # train_prob = train_prob / train_prob.sum(axis=-1).reshape((train_prob.shape[0], 1))
        train_prob = lasagne.layers.get_output(network, deterministic=False)
        loss = lasagne.objectives.categorical_crossentropy(train_prob, in_y).mean()
        if self.l2_reg > 0:
            loss += self.l2_reg * \
                lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
        self.params = lasagne.layers.get_all_params(network, trainable=True)

        if self.optimizer == 'sgd':
            updates = lasagne.updates.sgd(loss, self.params, learning_rate=self.learning_rate)
        elif self.optimizer == 'adam':
            updates = lasagne.updates.adam(loss, self.params)
        elif self.optimizer == 'rmsprop':
            updates = lasagne.updates.rmsprop(loss, self.params)
        elif self.optimizer == 'adagrad':
            updates = lasagne.updates.adagrad(loss, self.params, learning_rate=self.learning_rate)
        else:
            raise NotImplementedError('optimizer = %s' % self.optimizer)
        self.train_fn = theano.function([in_x, in_y], loss, updates=updates)

        test_prob = lasagne.layers.get_output(network, deterministic=True) * in_l
        pred = T.argmax(test_prob, axis=-1)
        acc = T.mean(T.eq(pred, in_y))
        self.test_fn = theano.function([in_x, in_l, in_y], acc)
        self.pred_fn = theano.function([in_x, in_l], pred)

    def legal_labels(self, stack, buf):
        labels = ([1] if len(stack) > 2 else [0]) * self.n_deprel
        labels += ([1] if len(stack) >= 2 else [0]) * self.n_deprel
        labels += [1] if len(buf) > 0 else [0]
        return labels

    def parse(self, eval_set, eval_batch_size=1000):
        ind = []
        steps = []
        stack = []
        buf = []
        arcs = []
        for i in xrange(len(eval_set)):
            n_words = len(eval_set[i]['word']) - 1
            ind.append(i)
            steps.append(n_words * 2)
            stack.append([0])
            buf.append([i + 1 for i in xrange(n_words)])
            arcs.append([])

        step = 0
        while len(ind) > 0:
            step = step + 1
            for mb in utils.get_minibatches(len(ind), eval_batch_size, shuffle=False):
                mb_x = [self.extract_features(stack[ind[k]], buf[ind[k]],
                                              arcs[ind[k]], eval_set[ind[k]]) for k in mb]
                mb_x = np.array(mb_x).astype('int32')
                mb_l = [self.legal_labels(stack[ind[k]], buf[ind[k]]) for k in mb]
                mb_l = np.array(mb_l).astype(_floatX)
                pred = self.pred_fn(mb_x, mb_l)
                for k, tran in zip(mb, pred):
                    i = ind[k]
                    if tran == self.n_trans - 1:
                        stack[i].append(buf[i][0])
                        buf[i] = buf[i][1:]
                    elif tran < self.n_deprel:
                        arcs[i].append((stack[i][-1], stack[i][-2], tran))
                        stack[i] = stack[i][:-2] + [stack[i][-1]]
                    else:
                        arcs[i].append((stack[i][-2], stack[i][-1], tran - self.n_deprel))
                        stack[i] = stack[i][:-1]
            ind = [k for k in ind if step < steps[k]]

        UAS = LAS = all_tokens = 0.0
        for i, ex in enumerate(eval_set):
            head = [-1] * len(ex['word'])
            label = [-1] * len(ex['word'])
            for h, t, l in arcs[i]:
                head[t] = h
                label[t] = l
            for pred_h, gold_h, pred_l, gold_l, pos in \
                    zip(head[1:], ex['head'][1:], label[1:], ex['label'][1:], ex['pos'][1:]):
                    assert self.id2tok[pos].startswith(P_PREFIX)
                    pos_str = self.id2tok[pos][len(P_PREFIX):]
                    if (self.with_punct) or (not utils.punct(self.language, pos_str)):
                        UAS += 1 if pred_h == gold_h else 0
                        LAS += 1 if (pred_h == gold_h) and (pred_l == gold_l) else 0
                        all_tokens += 1
        return UAS / all_tokens, LAS / all_tokens


def main(args):
    # Read examples
    logging.info('Load training data...')
    train_set = utils.read_conll(os.path.join(args.data_path, args.train_file),
                                 lowercase=args.lowercase,
                                 max_example=args.max_train)
    logging.info('Load development data...')
    dev_set = utils.read_conll(os.path.join(args.data_path, args.dev_file),
                               lowercase=args.lowercase)
    logging.info('-' * 100)

    nndep = Parser(train_set, args)
    logging.info('-' * 100)

    train_set = nndep.vectorize(train_set)
    dev_set = nndep.vectorize(dev_set)

    # Load embedding file
    if args.embedding_file is None:
        embeddings = {}
    else:
        logging.info('Load embedding file: %s' % args.embedding_file)
        embeddings = utils.get_embeddings(args.embedding_file)
        for w, w_emb in embeddings.items():
            assert len(w_emb) == args.embedding_size
        logging.info('#words = %d, dim = %d' % (len(embeddings), args.embedding_size))
        logging.info('-' * 100)

    # Build functions
    logging.info('Build functions...')
    if args.pre_trained is not None:
        dic = utils.load_params(args.pre_trained)
        pre_trained_params = dic['params']
        for i in nndep.id2tok:
            assert (i in dic['id2tok']) and (nndep.id2tok[i] == dic['id2tok'][i])
        logging.info('Load pre-trained model: %s' % args.pre_trained)
    else:
        pre_trained_params = None
    nndep.build_fn(embeddings, pre_trained_params)
    logging.info('Done.')

    logging.info('Initial testing...')
    UAS, LAS = nndep.parse(dev_set)
    logging.info('Dev UAS: %.2f, LAS: %.2f' % (UAS * 100.0, LAS * 100.0))

    # Create training and development instances
    logging.info('Create training instances...')
    train_examples = nndep.create_instances(train_set)
    logging.info('Create development instances...')
    dev_examples = nndep.create_instances(dev_set)
    logging.info('-' * 100)
    n_train = len(train_examples)
    n_dev = len(dev_examples)

    # Train
    logging.info('Start training...')
    start_time = time.time()
    n_updates = 0
    best_UAS = 0.0
    for epoch in range(args.n_epoches):
        minibatches = utils.get_minibatches(n_train, args.batch_size)
        for index, minibatch in enumerate(minibatches):
            train_x = np.array([train_examples[t][0] for t in minibatch]).astype('int32')
            train_l = np.array([train_examples[t][1] for t in minibatch]).astype(_floatX)
            train_y = [train_examples[t][2] for t in minibatch]

            train_loss = nndep.train_fn(train_x, train_y)
            logging.info('Epoch = %d, iter = %d (max. = %d), loss = %.2f, elapsed time = %.2f (s)' %
                         (epoch, index, len(minibatches), train_loss, time.time() - start_time))

            if train_loss != train_loss:
                raise Exception('train_loss is NaN.')
            n_updates += 1

            if n_updates % args.eval_iter == 0:
                size = min(n_train, n_dev)
                ind = np.random.choice(n_train, size, replace=False)
                all_acc = 0.0
                for mb in utils.get_minibatches(size, args.batch_size, shuffle=False):
                    train_x = np.array([train_examples[ind[t]][0] for t in mb]).astype('int32')
                    train_l = np.array([train_examples[ind[t]][1] for t in mb]).astype(_floatX)
                    train_y = [train_examples[ind[t]][2] for t in mb]
                    all_acc += nndep.test_fn(train_x, train_l, train_y) * len(mb)
                logging.info('Train accuracy: %.4f' % (all_acc / size))

                all_acc = 0.0
                for mb in utils.get_minibatches(n_dev, args.batch_size, shuffle=False):
                    dev_x = np.array([dev_examples[t][0] for t in mb]).astype('int32')
                    dev_l = np.array([dev_examples[t][1] for t in mb]).astype(_floatX)
                    dev_y = [dev_examples[t][2] for t in mb]
                    all_acc += nndep.test_fn(dev_x, dev_l, dev_y) * len(mb)
                logging.info('Dev accuracy:  %.4f' % (all_acc / n_dev))

                UAS, LAS = nndep.parse(dev_set)
                logging.info('Dev UAS: %.2f, LAS: %.2f' % (UAS * 100.0, LAS * 100.0))
                if UAS > best_UAS:
                    best_UAS = UAS
                    logging.info('Best UAS: epoch = %d, n_udpates = %d, UAS = %.2f, LAS = %.2f'
                                 % (epoch, n_updates, UAS * 100.0, LAS * 100.0))
                    if args.model_file is not None:
                        logging.info('Saving new model..')
                        utils.save_params(args.model_file, nndep.params,
                                          epoch=epoch,
                                          n_updates=n_updates,
                                          id2tok=nndep.id2tok,
                                          root_label=nndep.root_label)

if __name__ == '__main__':
    args = config.get_args()
    args.use_dep = args.use_dep and (not args.unlabeled)

    if args.job_id is not None:
        args.log_file = os.path.join(config.LOG_DIR, args.job_id + '.txt')
        args.model_file = os.path.join(config.MODEL_DIR, args.job_id + '.pkl.gz')
    else:
        args.log_file = None
        args.model_file = None

    np.random.seed(args.random_seed)
    lasagne.random.set_rng(np.random.RandomState(args.random_seed))

    if args.log_file is None:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M:%S')
    else:
        logging.basicConfig(filename=args.log_file,
                            filemode='w', level=logging.DEBUG,
                            format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M:%S')

    logging.info(' '.join(sys.argv))
    logging.info(args)
    logging.info('-' * 100)
    main(args)
