from __future__ import division
from __future__ import unicode_literals
import collections
import Levenshtein
import math

import utils

#@todo: add examples in the comments
# sequence based similarity measures

# levenshtein
@utils.check_args_for_none
@utils.check_strings_for_nulls
def levenshtein(string1, string2):
    """

    Args:
        string1, string2 (str): Input strings

    Returns:
        If string1 and string2 are valid strings then
            Levenshtein distance (float) between two strings is returned.
        else:
            NaN (from numpy) is returned
    Notes:
        This function internally uses python-levenshtein package

    """
    return Levenshtein.distance(string1, string2)

# jaro
@utils.check_args_for_none
@utils.check_strings_for_nulls
def jaro(string1, string2):
    """

    Args:
        string1, string2 (str): Input strings

    Returns:
        If string1 and string2 are valid strings then
            jaro similarity (float) between two strings is returned.
        else:
            NaN (from numpy) is returned
    Notes:
        This function internally uses python-levenshtein package
    """
    return Levenshtein.jaro(string1, string2)

# jaro-winkler
@utils.check_args_for_none
@utils.check_strings_for_nulls
def jaro_winkler(string1, string2, prefix_weight=0.1):
    """

    Args:
        string1, string2 (str): Input strings
        prefix_weight: Weight that should be given to the prefix (defaults to 0.1)

    Returns:
        If string1 and string2 are valid strings then
            jaro-winkler similarity (float) between two strings is returned.
        else:
            NaN (from numpy) is returned
    Notes:
        This function internally uses python-levenshtein package

    """
    return Levenshtein.jaro_winkler(string1, string2, prefix_weight)


# token based similarity measures

## set based similarity measures
@utils.check_args_for_none
@utils.check_tokens_for_nulls
def overlap(set1, set2):
    """

    Args:
        set1, set2 (set): Input sets.


    Returns:
        If set1 and set2 are valid sets/lists or single values then
            overlap similarity (float) between two sets is returned.
        else:
            NaN (from numpy) is returned

    """

    if not isinstance(set1, set):
        if not isinstance(set1, list):
            set1 = [set1]
        set1 = set(set1)
    if not isinstance(set2, set):
        if not isinstance(set1, list):
            set1 = [set1]
        set2 = set(set2)

    return float(len(set1 & set2))



## bag based similarity measures
@utils.check_args_for_none
@utils.check_tokens_for_nulls
def cosine(bag1, bag2):
    """
    Cosine similarity between two lists.
    There are multiple definitions of cosine similarity. This function implements ochihai coefficient.

    Args:
        bag1, bag2 (list): input lists

    Returns:
        If bag1 and bag2 are valid lists or single values then
            cosine similarity (float) between two sets is returned.
        else:
            NaN (from numpy) is returned

    """

    if not isinstance(bag1, list):
        bag1 = [bag1]
    if not isinstance(bag2, list):
        bag2 = [bag2]

    if len(bag1) == 0 or len(bag2) == 0:
        return 0.0

    c1 = collections.Counter(bag1)
    c2 = collections.Counter(bag2)

    c1_mag = sum(c1.values())
    c2_mag = sum(c2.values())

    int_mag = float(sum((c1&c2).values()))
    return int_mag/math.sqrt(c1_mag*c2_mag)







# hybrid similarity measures