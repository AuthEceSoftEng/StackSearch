### Brown Clusters
Contained in the `brown` directory in a `paths.pkl` pandas dataframe.

### Gazetteers
All files contained in the `gazetteers` directory.
Lists of Java APIs, Frameworks, Tools, Platforms etc.

### FastText Model
Contained in the `fasttext` directory.
1. Trained model.
2. GloVe style token vectors file.

### MiniBatch K-Means Clusters (trained on FastText vectors)
Contained in the `mbkm_clusters` directory.
1. Models contain cluster labels for each token of the trained FastText model.
2. Trained clusterers can be used for future token-vector calculations.

### Compound Embedding Features
Contained in the `ce_features` directory.
The dictionary of token frequencies `token_frequencies.json` was created using all the Stack Overflow Java posts.
The compound embedding features were calculated using the clusterers on the FastText token-vectors.

### Libraries (brown/fastText binaries)
1. `lib/wcluster`: used for brown clustering
2. `lib/fastText-0.1.0/fasttext`: used for fastText model training
