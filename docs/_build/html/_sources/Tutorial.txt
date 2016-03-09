Tutorial
========
Once the package is installed, the user can import the similarity functions and tokenizers
like this::

    from py_stringmatching import simfunctions, tokenizers

Further, the user can use the tokenizers and similarity functions like this::

    x = 'this is a string matching package for data science class'
    y = 'this string matching package can be used to generate features'
    f = simfunctions.cosine(tokenizers.whitespace(x), tokenizers.whitespace(y))


