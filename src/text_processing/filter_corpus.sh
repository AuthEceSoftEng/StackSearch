#!/bin/bash

show_usage() {
    echo -e "Usage: $0 [ARGUMENTS]"
    echo ""
    echo "arguments:"
    echo -e "  arg1 \tinput corpus file path"
    echo -e "  arg2 \toutput corpus file path [defaults to a '_nst' suffix if none is provided]"
}

remove_stacktrace() {
    sed -E "s/ at ([a-zA-Z_$][a-zA-Z0-9_$]*\.)+[a-zA-Z_$][a-zA-Z0-9_$]*\((([a-zA-Z_$][a-zA-Z0-9_$]{1,30}\.java:[0-9]{1,4})|([a-zA-Z]{1,30} [a-zA-Z]{1,30}))\)/ /g" |
    tr -s " "
}

remove_long_strings() {
    sed -E "s/([01]{4,20}\.){3,30}[01]{4,20}/ /g" | # remove concatenated binary strings
    sed -E "s/([a-zA-Z0-9_$]{2,50}\.){5,}[a-zA-Z0-9_$]{2,50}/ /g" | # remove long package names
    tr -s " "
}

if [ $# -eq 0 -o $# -gt 2 ]; then
    show_usage
    exit 1
fi

INPUTFILE=$1
DIR=$(dirname $INPUTFILE)
CORPUS=$(basename $INPUTFILE)
OUTPUTFILE=$DIR/$CORPUS\_nst

if [ $# -eq 2 ]; then
    OUTPUTFILE=$2
fi

cat $INPUTFILE | remove_stacktrace | remove_long_strings > $OUTPUTFILE