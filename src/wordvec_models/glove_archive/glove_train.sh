#!/bin/bash

VERSION=v0.1.2

CORPUS=train_data/corpus
VOCAB_FILE=glove_archive/vocab.txt
COOCCURRENCE_FILE=glove_archive/cooccurrence.bin
COOCCURRENCE_SHUF_FILE=glove_archive/cooccurrence.shuf.bin
LIB_DIR=glove_lib
SAVE_FILE=glove_archive/vectors_$VERSION
VERBOSE=2
MEMORY=5.0
VOCAB_MIN_COUNT=5
HEADER=1
VECTOR_SIZE=300
MAX_ITER=25
WINDOW_SIZE=10
MODEL=2
BINARY=0
NUM_THREADS=8
X_MAX=10
LEARNING_RATE=0.025

echo "OUTPUT_FILE: $SAVE_FILE"
#echo "$ $LIB_DIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
#$LIB_DIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
#echo "$ $LIB_DIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE"
#$LIB_DIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
#echo "$ $LIB_DIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
#$LIB_DIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
echo "$ $LIB_DIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -eta $LEARNING_RATE -iter $MAX_ITER -vector-size $VECTOR_SIZE -write-header $HEADER -binary $BINARY -model $MODEL -vocab-file $VOCAB_FILE -verbose $VERBOSE"
$LIB_DIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -eta $LEARNING_RATE -iter $MAX_ITER -vector-size $VECTOR_SIZE -write-header $HEADER -binary $BINARY -model $MODEL -vocab-file $VOCAB_FILE -verbose $VERBOSE
