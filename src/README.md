# Builder Scripts

## Corpus Builder

The **_Corpus Builder_** script makes use of the already built database in order to extract and group Post text data (Question Body, Answer Bodies, Comment Bodies).

1. Collects and groups text data.
2. Uses the post classifier and text evaluation metrics to keep or discard posts based on the level of noise.
3. Pre-processes the corpus (normalization and noise cleansing)
4. Builds the final corpus

## Model Trainer

The **_Model Trainer_** script is responsible for training the vector space models (or word embedding models) by making use of the previously build corpus.

- fastText model
- Tf-Idf model
- GloVe model (word vector index)

## Index Builder

The **_Index Builder_** script uses the previously built word embedding models as well as the post database in order to create: 
- The post-vector indices - essentially a vector representation of each post - for each type of word embedding model.
- The metadata lookup table (used by the presenter to print information and code snippets in a useful manner) which includes the following information: 

| Fields        | Information                                            |
| ------------- |:------------------------------------------------------:|
| PostId        | The Post Id                                            |
| Score         | The Post score                                         |
| Title         | The Post title                                         |
| ETags         | Post tags + extracted entities (from question/answers) |
| SnippetCount  | The number of code snippets found in the answers       |
| Snippets      | The code snippets found in the answers                 |

## Params

The `params.json` file is an easy way to configure the builder script options and file paths.

## Demo

The **_Demo_** script is used to test the final RSSE implementation, and provides three search model options:

- FastText
- Tf-Idf
- Hybrid