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
                ts = get_timestamp(line)
                update = get_update(line)
                last_ts = ts if ts is not None else last_ts
                last_update = update if update is not None else last_update

    if len(train_acc) > 0:
        k = np.argmax(train_acc)
        print 'job = %s, best train accuracy: %d / %d: %.4f' % (uid, k, len(train_acc), train_acc[k])

    if len(dev_acc) > 0:
        k = np.argmax(dev_acc)
        print 'job = %s, best dev accuracy: %d / %d: %.4f' % (uid, k, len(dev_acc), dev_acc[k])

    print 'last timestamp:', last_ts
    print 'last update: epoch = %s, iter = %s' % (last_update[0], last_update[1])
    print 'all jobs: %s' % (', '.join(all_jobs))
    return train_acc, dev_acc

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
        train_acc, dev_acc = read_log(uid)
        if not silent:
            plt.plot(range(len(train_acc)), train_acc, '.', color=colors[idx % len(colors)],
                     label=uid + ' (train)', alpha=0.3)
            plt.plot(range(len(dev_acc)), dev_acc, '-', color=colors[idx % len(colors)],
                     label=uid + ' (dev)')

    if not silent:
        plt.legend(loc='best')
        plt.show()
