#!/bin/bash

/lib/wcluster --text train_data/corpus_tok.txt --c 1000 --min-occur 4 --threads 8

/lib/fastText-0.1.0/fasttext skipgram -epoch 16 -dim 300 -minCount 8 -t 1e-6 -ws 10 -neg 8 -lr 0.025 -input train_data/corpus_tok.txt -output fasttext/fasttext_v0.1 -thread 8
