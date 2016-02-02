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
    This function calculates the hamming distance between the two equal length strings. It is the number of positions at
    which the corresponding symbols are different.

    Args:
        string1, string2 (str): Input strings

    Returns:
        If string1 and string2 are of same length the
            Hamming Distance distance (int) between two strings is returned.

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
@utils.sim_check_for_exact_match
@utils.sim_check_for_empty
def jaccard(set1, set2):
    """
    This function calculates the Jaccard similarity coefficient.
    The Jaccard coefficient measures similarity between finite sample sets, and is defined as the size of
    the intersection divided by the size of the union of the sample sets.

    Args:
        set1, set2 (set): Input sets.


    Returns:
        If set1 and set2 are valid sets/lists or single values then
            jaccard similarity (float) between two sets is returned.

    Examples:
        >>> jaccard(['data', 'science'], ['data'])
        0.5
        >>> jaccard({1, 1, 2, 3, 4}, {2, 3, 4, 5, 6, 7, 7, 8})
        0.375
        >>> jaccard(['data', 'management'], ['data', 'data', 'science'])
        0.3333333333333333
    """

    if not isinstance(set1, set):
        set1 = set(set1)
    if not isinstance(set2, set):
        set2 = set(set2)
    return float(len(set1 & set2)) / float(len(set1 | set2))


@utils.sim_check_for_none
@utils.sim_check_for_exact_match
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


@utils.sim_check_for_none
@utils.sim_check_for_exact_match
@utils.sim_check_for_empty
def tanimoto_coefficient(set1, set2):
    """
    This function calculates the Tanimoto coefficient.
    Calculates the Tanimoto coefficient over two sets. The similarity is defined as the cosine of the angle between the sets expressed as sparse vectors. Source: https://github.com/Simmetrics


    Args:
        set1, set2 (set): Input sets.


    Returns:
        If set1 and set2 are valid sets/lists or single values then
            Tanimoto coefficient (float) between two sets is returned.

    Examples:
        >>> tanimoto_coefficient(['data', 'science'], ['data'])
        0.7071067811865475
        >>> tanimoto_coefficient({1, 1, 2, 3, 4}, {2, 3, 4, 5, 6, 7, 7, 8})
        0.5669467095138409
        >>> tanimoto_coefficient(['data', 'management'], ['data', 'data', 'science'])
        0.4999999999999999
        >>> tanimoto_coefficient(['data', 'management'], ['data', 'management'])
        1.0
    """

    if not isinstance(set1, set):
        set1 = set(set1)
    if not isinstance(set2, set):
        set2 = set(set2)
    return float(len(set1 & set2)) / (math.sqrt(float(len(set1))) * math.sqrt(float(len(set2))))


# ---------------------- bag based similarity measures  ----------------------
@utils.sim_check_for_none
@utils.sim_check_for_exact_match
@utils.sim_check_for_empty
def cosine(bag1, bag2):
    """
    Cosine similarity between two lists.
    There are multiple definitions of cosine similarity. This function implements ochihai coefficient.

    Args:
        bag1, bag2 (list): input lists

    Returns:
        If bag1 and bag2 are valid lists or single values then
            cosine similarity (float) between two bags is returned.

    """

    c1 = collections.Counter(bag1)
    c2 = collections.Counter(bag2)

    c1_mag = sum(c1.values())
    c2_mag = sum(c2.values())

    int_mag = float(sum((c1 & c2).values()))
    return int_mag / math.sqrt(c1_mag * c2_mag)



# hybrid similarity measures
@utils.sim_check_for_none
@utils.sim_check_for_exact_match
@utils.sim_check_for_empty
def monge_elkan(bag1, bag2, sim_func=levenshtein):
    """
    Compute monge elkan similarity measures between two lists.
    Monge-Elkan measure is not symmetric. There is a variant that considers symmetricness.
    But currently we just implement the monge elkan measure as given in the DI book.

    Args:
        bag1, bag2 (list): Input lists
        sim_func: Similarity function to be used for each pair of tokens. This is expected to be a sequence-based
        similarity measure

    Returns:
        If bag1 and bag2 are valid lists then monge-elkan similarity between the two bags is returned.

    """
    sum_of_maxes = 0
    for t1 in bag1:
        max_sim = float('-inf')
        for t2 in bag2:
            max_sim = max(max_sim, sim_func(t1, t2))
        sum_of_maxes += max_sim
    sim = float(sum_of_maxes)/float(len(bag1))
    return sim


