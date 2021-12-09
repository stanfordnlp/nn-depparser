#!/usr/bin/env python

"""
    Replace the gold POS tags by the POS tags generated by Stanford
    POS tagger (http://nlp.stanford.edu/software/tagger.shtml).
    Usage: python replace_corenlp_pos.py language src.conll tgt.conll
"""

import sys
import os

TEXT_FILE = 'text.txt'
TAGGED_FILE = 'tagged.txt'


def get_tagger(language):
    if language == 'english' or language == 'english-wsj':
        return "edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger"
    elif language == 'chinese':
        return "edu/stanford/nlp/models/pos-tagger/chinese-distsim/chinese-distsim.tagger"
    elif language == 'german':
        return "edu/stanford/nlp/models/pos-tagger/german/german-hgc.tagger"
    elif language == 'french':
        return "edu/stanford/nlp/models/pos-tagger/french/french.tagger"
    elif language == 'spanish':
        return "edu/stanford/nlp/models/pos-tagger/spanish/spanish-distsim.tagger"
    else:
        return None

if __name__ == '__main__':
    if len(sys.argv) < 4:
        sys.exit('Usage: python replace_corenlp_pos.py language src.conll tgt.conll')

    language = sys.argv[1]
    tagger_model = get_tagger(language)
    if tagger_model is None:
        raise ValueError('Language %s is not supported.' % language)

    in_file = sys.argv[2]
    out_file = sys.argv[3]

    f_out = open(TEXT_FILE, 'w')
    n_sent = 0
    with open(in_file) as f_in:
        words = []
        for line in f_in.readlines():
            sp = line.strip().split('\t')
            if len(sp) == 10:
                if '-' not in sp[0]:
                    words.append(sp[1])
            elif len(words) > 0:
                n_sent += 1
                f_out.write('%s\n' % (' '.join(words)))
                words = []
        if len(words) > 0:
            n_sent += 1
            f_out.write('%s\n' % (' '.join(words)))
    f_out.close()

    print '#sent: %d' % n_sent
    print 'start tagging..'
    command = 'java edu.stanford.nlp.tagger.maxent.MaxentTagger \
          -sentenceDelimiter newline -tokenize false -outputFormat tsv \
          -model %s -textFile %s > %s' % (tagger_model, TEXT_FILE, TAGGED_FILE)
    command = ' '.join(command.split())
    print 'command: %s' % command
    os.system(command)

    tagged = []
    with open(TAGGED_FILE) as f:
        for line in f.readlines():
            sp = line.strip().split('\t')
            if len(sp) == 2:
                tagged.append((sp[0], sp[1]))
    print '#tokens: %d' % len(tagged)

    f_out = open(out_file, 'w')
    k = 0
    with open(in_file) as f_in:
        for line in f_in.readlines():
            sp = line.strip().split('\t')
            if (len(sp) == 10) and ('-' not in sp[0]):
                #assert sp[1] == tagged[k][0]
                sp[4] = tagged[k][1]
                k += 1
                f_out.write('%s\n' % '\t'.join(sp))
            else:
                f_out.write('%s\n' % line.strip())
    f_out.close()

    os.remove(TEXT_FILE)
    os.remove(TAGGED_FILE)