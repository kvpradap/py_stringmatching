from __future__ import division
from __future__ import unicode_literals
import collections
import Levenshtein
import math

import utils


# @todo: add examples in the comments
# ---------------------- sequence based similarity measures  ----------------------

# jaro
@utils.sim_check_for_none
@utils.sim_check_for_empty
def jaro(string1, string2):
    """

    Args:
        string1, string2 (str): Input strings

    Returns:
        If string1 and string2 are valid strings then
            jaro similarity (float) between two strings is returned.

    Notes:
        This function internally uses python-levenshtein package
    """
    return Levenshtein.jaro(string1, string2)


# jaro-winkler
@utils.sim_check_for_none
@utils.sim_check_for_empty
def jaro_winkler(string1, string2, prefix_weight=0.1):
    """

    Args:
        string1, string2 (str): Input strings
        prefix_weight: Weight that should be given to the prefix (defaults to 0.1)

    Returns:
        If string1 and string2 are valid strings then
            jaro-winkler similarity (float) between two strings is returned.

    Notes:
        This function internally uses python-levenshtein package

    """
    return Levenshtein.jaro_winkler(string1, string2, prefix_weight)


# levenshtein
@utils.sim_check_for_same_len
def hamming_distance(string1, string2):
    """
    Args:
        string1, string2 (str): Input strings

    Returns:
        If string1 and string2 are of same length the
            Hamming Distance distance (int) between two strings is returned.

    Notes:
        This function has same implementation as hammingDistance in Python3

    Examples:
        >>> hamming_distance('', '')
        0
        >>> hamming_distance('alex', 'john')
        4
        >>> hamming_distance(' ', 'a')
        0
        >>> hamming_distance('JOHN', 'john')
        4
    """
    return sum(bool(ord(c1) - ord(c2)) for c1, c2 in zip(string1, string2))


@utils.sim_check_for_none
def levenshtein(string1, string2):
    """

    Args:
        string1, string2 (str): Input strings

    Returns:
        If string1 and string2 are valid strings then
            Levenshtein distance (float) between two strings is returned.

    Notes:
        This function internally uses python-levenshtein package

    """
    return Levenshtein.distance(string1, string2)


# ---------------------- token based similarity measures  ----------------------

# ---------------------- set based similarity measures  ----------------------

@utils.sim_check_for_none
@utils.sim_check_for_empty
def jaccard(set1, set2):
    """

    Args:
        set1, set2 (set): Input sets.


    Returns:
        If set1 and set2 are valid sets/lists or single values then
            jaccard similarity (float) between two sets is returned.
    Notes:
        This function calculate the Jaccard similarity and not Jaccard distance.

    Examples:
        >>> jaccard(['data', 'science'], ['data'])
        0.5
        >>> jaccard({1, 1, 2, 3, 4}, {2, 3, 4, 5, 6, 7, 7, 8})
        0.375
        >>> jaccard(['data', 'management'], ['data', 'data', 'science'])
        0.3333333333333333
    """
    set1 = set(set1) if not isinstance(set1, set) else set1
    set2 = set(set2) if not isinstance(set2, set) else set2
    return float(len(set1 & set2)) / float(len(set1 | set2))


@utils.sim_check_for_none
@utils.sim_check_for_empty
def overlap(set1, set2):
    """

    Args:
        set1, set2 (set): Input sets.


    Returns:
        If set1 and set2 are valid sets/lists or single values then
            overlap similarity (float) between two sets is returned.
\
    """
    if not isinstance(set1, set):
        set1 = set(set1)
    if not isinstance(set2, set):
        set2 = set(set2)

    return float(len(set1 & set2))/min(len(set1), len(set2))


# ---------------------- bag based similarity measures  ----------------------
@utils.sim_check_for_none
@utils.sim_check_for_empty
def cosine(bag1, bag2):
    """
    Cosine similarity between two lists.
    There are multiple definitions of cosine similarity. This function implements ochihai coefficient.

    Args:
        bag1, bag2 (list): input lists

    Returns:
        If bag1 and bag2 are valid lists or single values then
            cosine similarity (float) between two sets is returned.

    """

    c1 = collections.Counter(bag1)
    c2 = collections.Counter(bag2)

    c1_mag = sum(c1.values())
    c2_mag = sum(c2.values())

    int_mag = float(sum((c1 & c2).values()))
    return int_mag / math.sqrt(c1_mag * c2_mag)



    # hybrid similarity measures
