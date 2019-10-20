#!/bin/bash

show_usage() {
    echo -e "Usage: $0 [ARGUMENTS]"
    echo ""
    echo "arguments:"
    echo -e "  arg1 \tinput corpus file path"
    echo -e "  arg2 \toutput corpus file path [defaults to a '_norm' suffix if none is provided]"
}

normalize_text() {
    sed -E 's/[^ \.]{40,}/ /g' | sed -E 's/[[:punct:]]{4,}/ /g' | 
    sed -E -e 's/[!"%&$^*@`~(),?=\/\|<:;>\{\}-]/ /g' -e 's/\[/ /g' -e 's/\]/ /g' | 
    sed -E -e 's/([Cc]#)|#/\1/g' -e 's/([Cc]\+\+)|\+/\1/g' | tr -s ' '
}

if [ $# -eq 0 -o $# -gt 2 ]; then
    show_usage
    exit 1
fi

INPUTFILE=$1
DIR=$(dirname $INPUTFILE)
CORPUS=$(basename $INPUTFILE)
OUTPUTFILE=$DIR/$CORPUS\_norm

if [ $# -eq 2 ]; then
    OUTPUTFILE=$2
fi

cat $INPUTFILE | normalize_text > $OUTPUTFILE