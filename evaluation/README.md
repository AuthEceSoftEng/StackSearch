# Evaluation

The evaluation process of our search models was done in the following two stages.

## Experiments

To conduct the experiments 18 queries were used. Groups of those queries present the same question expressed with different terms in order to show the semantic value added by the fastText word vector model.  

| ID | Query                                     | Group |
| -- |:-----------------------------------------:| ----- |
| 1  | How to read a comma separated file?       |   1   |
| 2  | How to read a CSV file?                   |   1   |
| 3  | How to read a delimited file?             |   1   |
| 4  | How to read input from console?           |   2   |
| 5  | How to read input from terminal?          |   2   |
| 6  | How to read input from command prompt?    |   2   |
| 7  | How to play an mp3 file?                  |   3   |
| 8  | How to play an audio file?                |   3   |
| 9  | How to compare dates?                     |   4   |
| 10 | How to compare time stamps?               |   4   |
| 11 | How to dynamically load a class?          |   5   |
| 12 | How to load a jar/class at runtime?       |   5   |
| 13 | How to calculate checksums for files?     |   6   |
| 14 | How to calculate MD5 checksums for files? |   6   |
| 15 | How to iterate through a hashmap?         |   7   |
| 16 | How to loop over a hashmap?               |   7   |
| 17 | How to split a string?                    |   8   |
| 18 | How to handle an exception?               |   9   |

The metrics used to evaluate our search models were Average Precision (AP), Mean Average Precision (MAP), Mean Search Length (MSL).

## PostLink Experiments

In order to evaluate our search models without bias we used the PostLink experiment. In Stack Overflow, the presence of a link between two questions is an indicator that the two questions are similar.  
_Note, of course, that the opposite assumption, i.e. that any two questions that are not linked are not similar to each other, is not necessarily correct._

Our link evaluation dataset contained roughly 200k question posts (as opposed to the original dataset that had approximately 1.3 million question posts).  
Question posts were selected using certain thresholds for performance and quality reasons (`score >= -3 && number_of_snippets >= 1`).  
These question posts had approximately 37000 links.

## Visualization

Some early visualization experiments were done in order to assess the value of using a word vector model in our search algorithm. Post Title and Body vectors (dimensionality of 300) produced by one of our early fastText models, were fed into the [t-SNE algorithm](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding).  
By using the Post Tags to evaluate post similarity we got the following results.  

![t-SNE Graph / Title Vectors](visualizations/[t-SNE]_TitleVectors.png?raw=true)  

![t-SNE Graph / Body Vectors](visualizations/[t-SNE]_BodyVectors.png?raw=true)  

In both cases we observe certain clear clusters being formed which hints to the semantic value added by the fastText model.