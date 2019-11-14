# StackSearch
RSSE implementation using crowd-sourced data from Stack Overflow.  
Utilizes word embedding models trained on a custom corpus, to recommend code snippets and/or Stack Overflow posts to user queries.

This project focuses on an RSSE for Java queries and makes use of the StackOverflow data dump.

## Environment Setup

1. Clone [StackSearch](https://github.com/AuthEceSoftEng/StackSearch) repository and run `make setup`. Handles virtual env setup and dependency installation.

   ```sh
   git clone git@github.com:AuthEceSoftEng/StackSearch.git ${HOME}/stacksearch && cd ${HOME}/stacksearch/src && make setup
   ```

2. Source the activate file to get into the python virtual environment.

   ```sh
   source ${HOME}/.stacksearch/bin/activate
   ```

## Usage

```
demo.py [-h] {fasttext,tfidf,hybrid} ...

StackSearch Demo

positional arguments:
  {fasttext,tfidf,hybrid}
    fasttext            Use a FastText model for searching.
    tfidf               Use a TF-IDF model for searching.
    hybrid              Use a Hybrid model (FastText & TF-IDF) for searching.

optional arguments:
  -h, --help            show this help message and exit

Search model 'fasttext'
usage: demo.py fasttext [-h] MODEL INDEX METADATA RESULTS

positional arguments:
  MODEL       Path to the FastText model.
  INDEX       Path to the FastText search index.
  METADATA    Path to the metadata index.
  RESULTS     Number of results for each query.

optional arguments:
  -h, --help  show this help message and exit


Search model 'tfidf'
usage: demo.py tfidf [-h] MODEL INDEX METADATA RESULTS

positional arguments:
  MODEL       Path to the TF-IDF model.
  INDEX       Path to the TF-IDF search index.
  METADATA    Path to the metadata index.
  RESULTS     Number of results for each query.

optional arguments:
  -h, --help  show this help message and exit


Search model 'hybrid'
usage: demo.py hybrid [-h]
                      FASTTEXT MODEL TFIDF MODEL FASTTEXT INDEX TFIDF INDEX
                      METADATA RESULTS

positional arguments:
  FASTTEXT MODEL  Path to the FastText model.
  TFIDF MODEL     Path to the TF-IDF model.
  FASTTEXT INDEX  Path to the FastText search index.
  TFIDF INDEX     Path to the TF-IDF search index.
  METADATA        Path to the metadata index.
  RESULTS         Number of results for each query.

optional arguments:
  -h, --help      show this help message and exit
```

## Example

```sh
./demo.py hybrid wordvec_models/fasttext_archive/ft_v0.6.1.bin wordvec_models/tfidf_archive/tfidf_v0.3.pkl wordvec_models/index/ft_v0.6.1_post_index.pkl wordvec_models/index/tfidf_v0.3_post_index.pkl wordvec_models/index/extended_metadata.pkl 20
```

#### Preview

```
Hybrid model

FastText model ft_v0.6.1.bin loaded.
TF-IDF model tfidf_v0.3.pkl loaded.
Index keys used: TitleV, BodyV

Query [query + enter], quit ['q' + enter]: How to calculate md5 checksums?
Tags (e.g. java, android): md5

1/20
################################# CODE #################################

DigestUtils.md5Hex(str);

################################# CODE #################################

Title: Java calculate MD5 hash
Post: https://stackoverflow.com/questions/7776116
Answer: https://stackoverflow.com/questions/7776244

Answer score: 38
Snippets for this post: 3
Top 8 tags for this query: java, md5, messagedigest, android, spring, checksum, md5sum, hashcode

Next code snippet [enter], new query ['q' + enter]:
```