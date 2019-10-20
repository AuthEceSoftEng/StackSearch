#!/bin/bash

echo "fasttext skipgram -epoch 24 -dim 300 -t 1e-6 -ws 10 -neg 10 -lr 0.025 -input train_data/corpus_norm_v0.3 -output fasttext_archive/ft_v0.6.1 -thread 8 -saveOutput"
lib/fastText-0.1.0/fasttext skipgram -epoch 24 -dim 300 -t 1e-6 -ws 10 -neg 10 -lr 0.025 -input train_data/corpus_norm_v0.3 -output fasttext_archive/ft_v0.6.1 -thread 8 -saveOutput
