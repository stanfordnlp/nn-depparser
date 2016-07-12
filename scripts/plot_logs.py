#!/usr/bin/env python

import sys
import numpy as np


def get_timestamp(line):
    line = line.strip()
    if line.startswith('03-') or line.startswith('04-') or \
       line.startswith('05-') or line.startswith('06-') or \
       line.startswith('07-'):
            return ' '.join(line.split(' ')[:2])
    return None


def get_update(line):
    line = line.strip()
    return (line.split(' ')[4][:-1], line.split(' ')[7]) if ('Epoch = ' in line) and ('iter =' in line) else None


def read_log(uid):
    train_acc = []
    dev_acc = []
    UAS = []
    all_jobs = []

    # Read head file to check if there is a pre-trained model
    current_uid = uid
    while current_uid is not None:
        all_jobs.append(current_uid)
        prev_uid = None
        with open('../logs/' + current_uid + '.txt') as f:
            for i in range(500):
                line = f.readline().strip()
                if 'pre-trained model' in line:
                    prev_uid = line.split(' ')[-1]
                    break
        current_uid = prev_uid
    all_jobs.reverse()

    last_ts = None
    last_update = None
    for cuid in all_jobs:
        with open('../logs/' + cuid + '.txt') as f:
            for line in f.readlines():
                sline = line.strip()
                if 'Train acc' in sline:
                    train_acc.append(float(sline.split(' ')[-1]))
                if 'Dev acc' in sline:
                    dev_acc.append(float(sline.split(' ')[-1]))
                if 'UAS' in sline:
                    UAS.append(float(sline.split(' ')[-1]))
                ts = get_timestamp(line)
                update = get_update(line)
                last_ts = ts if ts is not None else last_ts
                last_update = update if update is not None else last_update

    print '-' * 20, uid, '-' * 20
    if len(train_acc) > 0:
        k = np.argmax(train_acc)
        print 'best train accuracy: %d / %d: %.4f' % (k, len(train_acc), train_acc[k])

    if len(dev_acc) > 0:
        k = np.argmax(dev_acc)
        print 'best dev accuracy: %d / %d: %.4f' % (k, len(dev_acc), dev_acc[k])

    if len(UAS) > 0:
        print 'best UAS : %.4f' % max(UAS)

    print 'last timestamp:', last_ts
    print 'last update: epoch = %s, iter = %s' % (last_update[0], last_update[1])
    print 'all jobs: %s' % (', '.join(all_jobs))
    return train_acc, dev_acc, UAS

if __name__ == '__main__':
    argv = sys.argv
    silent = len(argv) > 1 and argv[1] == '-silent'
    argv = argv[2:] if silent else argv[1:]

    if len(argv) < 1:
        print'Usage: python plot_logs.py [-silent] <uid_1> <uid_2>, ...'
        exit(1)

    if not silent:
        import matplotlib.pyplot as plt

    jobs = sorted(argv)
    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
    for (idx, uid) in enumerate(jobs):
        train_acc, dev_acc, dev_UAS = read_log(uid)
        if not silent:
            plt.plot(range(len(train_acc)), train_acc, '.', color=colors[idx % len(colors)],
                     label=uid + ' (train)', alpha=0.3)
            plt.plot(range(len(dev_acc)), dev_acc, '-', color=colors[idx % len(colors)],
                     label=uid + ' (dev)')
            plt.plot(range(len(dev_UAS)), dev_UAS, '--', color=colors[idx % len(colors)],
                     label=uid + ' (dev UAS)')

    if not silent:
        plt.legend(loc='best')
        plt.show()
