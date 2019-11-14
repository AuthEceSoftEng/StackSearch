# Text Processing Utilities

### Scripts

`utils.py`: Provides a number of utility function used throughout the project for text pre-processing and corpus building.  
`tokenizer.py`: A modified [spaCy tokenizer](https://spacy.io/api/tokenizer "spaCy Tokenizer") that is build to respect API call structure (method brackets, nested calls etc.) as well as API related terminology.  
`text_eval.py`: Provides some utility functions and a text/post evaluation function created to calculate text/post quality (noise or not) based on certain hard set metrics. Metric thresholds were set after experimentation.

### spaCy

[spaCy](https://spacy.io "spaCy") is a great NLP library that provides a multiltude of tools that excel in accuracy and performance. In the current project we used a modified version of their tokenizer.
