# Vector Space Models

All vector space models were trained on an extensively pre-processed corpus from StackOverflow Java Posts (questions, answers, comments).
- Each search model provides an `infer_vector` function which is responsible for producing the vector representation of the user query.
- Each search model extends the `BaseSearchModel` class found in `search_model.py` which includes a ranking function. The ranking function is based on cosine similarities calculated using the query vector and every post vector found in the given index.
- Additionaly the `BaseSearchModel` provides the necessary functions to calculate batch cosine similarities as well as a `presenter` function which prints the `ranking` function's results in a useful manner (making use of the metadata produced by the **Index Builder**).

## FastText

The [fastText](https://fasttext.cc/) search model produced very relevant results in our experiments showing its power in query semantic similarity (e.g. Read CSV file. Read delimited file.).  
Although the vectors produced by large text bodies presented worse performance than this of the Tf-Idf model (limited vector dimensions e.g. 300) its ngram utility proved to be a great advantage in query misspellings.

## Tf-Idf

The [Tf-Idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) search model has been used extensively in multitude of information retrieval projects and search engines and produces great results with efficiency.

## Hybrid

The Hybrid search model uses a combination of the Tf-Idf and fastText models to infer sentence vectors. It uses by default a formula of `sentence_vector = 0.5*tfidf_vector + 0.5*fasttext_vector`.  
This model performed the best in our experiments confirming our hypothesis that semantic and statistical models can be combined to produce a performant RSSE.

## GloVe

The GloVe search model infers word and sentence vectors using the word vector dictionary produced during training. It calculates sentence vectors by mimiking the fastText algorithm (average of the unit norm vectors of every token).  
While GloVe word vectors have been used in the past with great success, in this project it had a poor performance. This performance can be attributed to the nature and informality of online speech as well as the programming terminology and API calls found in the text.

# Index

The indices produced by the `index_builder.py` script provide a post-vector lookup table in order to calculate cosine similarities with the user given queries.  
The metadata files include useful information to be presented when a query is issued. When the ranking function returns the top relevant results code snippets and additional useful information is presented alongside them.
