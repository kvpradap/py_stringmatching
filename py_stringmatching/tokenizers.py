from .compat import _range
from py_stringmatching import utils

#@todo: add examples in the comments
@utils.tok_check_for_none
@utils.tok_check_for_string_input
def qgram(input_string, qval=2):
    """
    QGram tokenizer.  A q-gram is defined as all sequences of q characters. Q-grams are also known as n-grams and
    k-grams.

    Args:
        input_string (str): A string to extract q-grams from

        qval (int): The q-gram length (defaults to 2)

    Returns:
        If the input string is not None and qval is less than length of the input  then,
        a list of qgrams is returned.

    """
    qgram_list = []

    if len(input_string) < qval or qval < 1:
        return qgram_list

    qgram_list = [input_string[i:i+qval] for i in _range(len(input_string)-(qval-1))]
    return qgram_list

@utils.tok_check_for_none
@utils.tok_check_for_string_input
def delimiter(input_string, delim_str=' '):
    """
    Delimiter based tokenizer

    Args:
        input_string (str): A string to extract tokens from.

        delim_str (str): Delimiter string


    Returns:
        If the input string is not None and delim_str is not None, then a list of tokens are returned.

    """
    return input_string.split(delim_str)

@utils.tok_check_for_none
@utils.tok_check_for_string_input
def whitespace(input_string):
    """
    White space based tokenizer

    Args:
        input_string (str): A string to extract tokens from.

    Returns:
        If the input string is not None, then a list of tokens are returned.

    """
    return input_string.split()


