import collections
from py_stringmatching import tokenizers, simfunctions
x = 'this is a good one'
y = 'this is a bad one'
z = 'this is the worst one'
x_tok, y_tok, z_tok = tokenizers.whitespace(x), tokenizers.whitespace(y), tokenizers.whitespace(z)

def tfidf(x_tok, y_tok, corpus_list = None):
    if corpus_list is None:
        corpus_list = [x_tok, y_tok]
    x_tf = collections.Counter(x_tok)
    y_tf = collections.Counter(y_tok)
    num_docs = len(corpus_list)


    pass


