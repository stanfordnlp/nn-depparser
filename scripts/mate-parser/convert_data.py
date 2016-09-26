#! /usr/bin/python

'''
Convert CoNLL-X format to CoNLL-2009 format
'''

import sys;

if len(sys.argv) != 2:
    sys.stderr.write('Usage: python convert_data.py <input_file> > <output_file>\n')
    sys.exit(1)

# Open File
f = open(sys.argv[1],'rt');
wrds = ""; pos = ""; labs = ""; par = "";
for line in f:
    sent = line.split();
    if len(sent) > 0:
        print sent[0] + "\t" + sent[1] + "\t_\t_\t" + sent[4] + "\t_\t_\t_\t" + sent[6] + "\t_\t" + sent[7] + "\t_\t_\t_";
    else:
        print "";
f.close();

