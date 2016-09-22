#!/usr/bin/env python

import glob

DATA_DIR = '/scr/nlp/data/ldc/LDC2016T13-ctb9.0/data/bracketed/'
SPLIT_FILE = '/scr/nlp/data/ldc/LDC2016T13-ctb9.0/docs/ctb9.0-file-list.txt'

if __name__ == '__main__':

    mapping = {}
    files = glob.glob(DATA_DIR + '/chtb*')
    for f in files:
        name = f.split('/')[-1].split('.')[0]
        mapping[name] = f

    f_out = {}
    for split in ['train', 'dev', 'test']:
        f_out[split] = open(split + '.mrg', 'w')
    with open(SPLIT_FILE) as f_in:
        for line in f_in.readlines():
            sp = line.strip().split(' ')
            if sp[0].startswith('chtb'):
                split = 'train'
                name = sp[0]
            else:
                name = sp[1]
                split = 'dev' if sp[0] == '!' else 'test'

            with open(mapping[name]) as f:
                for line in f.readlines():
                    f_out[split].write(line.strip() + '\n')

    for split in ['train', 'dev', 'test']:
        f_out[split].close()
