#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo 'Usage: ./test.sh <model_file> <test_file> <output_file>'
    exit 1
fi

java -Xmx10G -classpath anna-3.61.jar is2.parser.Parser -model $1 -test $2 -out $3
