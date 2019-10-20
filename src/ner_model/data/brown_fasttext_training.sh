#!/bin/bash

#python mytokenizer.py /run/media/ncode/C8FC7880FC786B18/stacksearch/src/data/init_corpus_nst corpus_tok

#./wcluster --text corpus_tok --c 1000 --min-occur 8 --threads 8

../../lib/fastText-0.1.0/fasttext skipgram -epoch 16 -dim 300 -minCount 8 -t 1e-6 -ws 10 -neg 8 -lr 0.025 -input corpus_tok -output fasttext_v0.1 -thread 8
show_usage() {
    echo -e "Usage: $0 [ARGUMENTS]"
    echo ""
    echo "arguments:"
    echo -e "  arg1 \tinput corpus file path"
}

normalize_text() {
    sed -E 's/[^ \.]{40,}/ /g' | sed -E 's/[[:punct:]]{4,}/ /g' | 
    sed -E -e 's/[!"%&$^*@`~(),?=\/\|<:;>\{\}-]/ /g' -e 's/\[/ /g' -e 's/\]/ /g' | 
    sed -E -e 's/([Cc]#)|#/\1/g' -e 's/([Cc]\+\+)|\+/\1/g' | tr -s ' '
}

if [ $# -ne 1 ]; then
    show_usage
    exit 1
fi

INPUTFILE=$1
DIR=$(dirname $INPUTFILE)
CORPUS=$(basename $INPUTFILE)

BROWN_OUTPUT=
OUTPUTFILE=$DIR/$CORPUS\_norm


cat $INPUTFILE | normalize_text > $OUTPUTFILE