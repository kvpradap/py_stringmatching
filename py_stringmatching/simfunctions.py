# coding=utf-8
from __future__ import division
from __future__ import unicode_literals

import Levenshtein
import math

from py_stringmatching import utils
import numpy as np
from .compat import _range


def sim_ident(s1, s2):
    return int(s1 == s2)

# ---------------------- sequence based similarity measures  ----------------------


@utils.sim_check_for_none
@utils.tok_check_for_string_input
@utils.sim_check_for_empty
def affine(string1, string2, gap_start=1, gap_continuation=0.5, sim_score=sim_ident):
    """
    Computes the Affine gap score between two strings.

    The Affine gap measure is an extension of the Needleman-Wunsch measure that handles the longer gaps more
    gracefully.

    For more information refer to string matching chapter in the DI book.

    Args:
        string1,string2 (str) : Input strings

        gap_start (float): Cost for the gap at the start (defaults to 1)

        gap_continuation (float) : Cost for the gap continuation (defaults to 0.5)

        sim_score (function) : Function computing similarity score between two chars, represented as strings
            (defaults to identity).

    Returns:
        Affine gap score (float)

    Raises:
        TypeError : If the inputs are not strings

    Examples:
        >>> affine('dva', 'deeva')
        1.5
        >>> affine('dva', 'deeve', gap_start=2, gap_continuation=0.5)
        -0.5
        >>> affine('AAAGAATTCA', 'AAATCA', gap_continuation=0.2, sim_score=lambda s1, s2 : (int(1 if s1 == s2 else 0)))
        4.4
    """
    gap_start = -gap_start
    gap_continuation = -gap_continuation
    M = np.zeros((len(string1) + 1, len(string2) + 1), dtype=np.float)
    X = np.zeros((len(string1) + 1, len(string2) + 1), dtype=np.float)
    Y = np.zeros((len(string1) + 1, len(string2) + 1), dtype=np.float)

    for i in _range(1, len(string1) + 1):
        M[i][0] = -float("inf")
        X[i][0] = gap_start + (i - 1) * gap_continuation
        Y[i][0] = -float("inf")

    for j in _range(1, len(string2) + 1):
        M[0][j] = -float("inf")
        X[0][j] = -float("inf")
        Y[0][j] = gap_start + (j - 1) * gap_continuation

    for i in _range(1, len(string1) + 1):
        for j in _range(1, len(string2) + 1):
            M[i][j] = sim_score(string1[i - 1], string2[j - 1]) + max(M[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1])
            X[i][j] = max(gap_start + M[i - 1][j], gap_continuation + X[i - 1][j])
            Y[i][j] = max(gap_start + M[i][j - 1], gap_continuation + Y[i][j - 1])
    return max(M[len(string1)][len(string2)], X[len(string1)][len(string2)], Y[len(string1)][len(string2)])


# jaro
@utils.sim_check_for_none
@utils.tok_check_for_string_input
@utils.sim_check_for_empty
def jaro(string1, string2):
    """
    Computes the Jaro measure between two strings.

    The Jaro measure is a type of edit distance, This was developed mainly to compare short strings,
    such as first and last names.


    Args:
        string1,string2 (str): Input strings

    Returns:
        Jaro measure (float)


    Raises:
        TypeError : If the inputs are not strings


    Examples:
        >>> jaro('MARTHA', 'MARHTA')
        0.9444444444444445
        >>> jaro('DWAYNE', 'DUANE')
        0.8222222222222223
        >>> jaro('DIXON', 'DICKSONX')
        0.7666666666666666


    """

    len_s1 = len(string1)
    len_s2 = len(string2)

    max_len = max(len_s1, len_s2)
    search_range = (max_len // 2) - 1
    if search_range < 0:
        search_range = 0

    flags_s1 = [False] * len_s1
    flags_s2 = [False] * len_s2

    common_chars = 0
    for i, ch_s1 in enumerate(string1):
        low = i - search_range if i > search_range else 0
        hi = i + search_range if i + search_range < len_s2 else len_s2 - 1
        for j in _range(low, hi + 1):
            if not flags_s2[j] and string2[j] == ch_s1:
                flags_s1[i] = flags_s2[j] = True
                common_chars += 1
                break
    if not common_chars:
        return 0

    k = trans_count = 0
    for i, f_s1 in enumerate(flags_s1):
        if f_s1:
            for j in _range(k, len_s2):
                if flags_s2[j]:
                    k = j + 1
                    break
            if string1[i] != string2[j]:
                trans_count += 1
    trans_count /= 2
    # print trans_count, common_chars


    common_chars = float(common_chars)
    weight = ((common_chars / len_s1 + common_chars / len_s2 +
               (common_chars - trans_count) / common_chars)) / 3
    return weight


# jaro-winkler
@utils.sim_check_for_none
@utils.tok_check_for_string_input
@utils.sim_check_for_empty
def jaro_winkler(string1, string2, prefix_weight=0.1):
    """
    Computes the Jaro-Winkler measure between two strings.

    The Jaro-Winkler measure is designed to capture cases where two strings have a low Jaro score, but share a prefix
    and thus are likely to match.


    Args:
        string1,string2 (str): Input strings

        prefix_weight (float): Weight to give the prefix (defaults to 0.1)

    Returns:
        Jaro-Winkler measure (float)

    Raises:
        TypeError : If the inputs are not strings


    Examples:
        >>> jaro_winkler('MARTHA', 'MARHTA')
        0.9611111111111111
        >>> jaro_winkler('DWAYNE', 'DUANE')
        0.84
        >>> jaro_winkler('DIXON', 'DICKSONX')
        0.8133333333333332

    """
    jw_score = jaro(string1, string2)
    max_len = max(len(string1), len(string2))
    j = min(max_len, 4)
    i = 0
    while i < j and string1[i] == string2[i] and string1[i]:
        i += 1
    if i:
        jw_score += i * prefix_weight * (1 - jw_score)

    return jw_score


@utils.sim_check_for_none
@utils.tok_check_for_string_input
@utils.sim_check_for_same_len
def hamming_distance(string1, string2):
    """
    Computes the Hamming distance between two strings.

    The Hamming distance between two strings of equal length is the number of positions at which the corresponding
    symbols are different. In another way, it measures the minimum number of substitutions required to change
    one string into the other, or the minimum number of errors that could have transformed one string into the other.


    Args:
        string1,string2 (str): Input strings

    Returns:
        Hamming distance (int)

    Raises:
        TypeError : If the inputs are not strings
        ValueError : If the input strings are not of same length


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
@utils.sim_check_for_string_inputs
def levenshtein(string1, string2):
    """
    Computes the Levenshtein distance between two strings.

    Levenshtein distance computes the minimum cost of transforming one string into the other. Transforming a string
    is carried out using a sequence of the following operators: delete a character, insert a character, and
    substitute one character for another.

    Args:
        string1,string2 (str): Input strings

    Returns:
        Levenshtein distance (int)

    Raises:
        TypeError : If the inputs are not strings

    Examples:
        >>> levenshtein('a', '')
        1
        >>> levenshtein('example', 'samples')
        3
        >>> levenshtein('levenshtein', 'frankenstein')
        6


    Note:
        This implementation internally uses python-levenshtein package to compute the Levenshtein distance

    """
    return Levenshtein.distance(string1, string2)


@utils.sim_check_for_none
@utils.sim_check_for_string_inputs
def needleman_wunsch(string1, string2, gap_cost=1.0, sim_score=sim_ident):
    """
    Computes the Needleman-Wunsch measure between two strings.

    The Needleman-Wunsch generalizes the Levenshtein distance and considers global alignment between two strings.
    Specifically, it is computed by assigning a score to each alignment between two input strings and choosing the
    score of the best alignment, that is, the maximal score.

    An alignment between two strings is a set of correspondences between the characters of between them, allowing for
    gaps.

    Args:
        string1,string2 (str) : Input strings

        gap_cost (float) : Cost of gap (defaults to 1.0)

        sim_score (function) : Similarity function to give a score for the correspondence between characters. Defaults
            to an identity function, where if two characters are same it returns 1.0 else returns 0.


    Returns:
        Needleman-Wunsch measure (float)


    Raises:
        TypeError : If the inputs are not strings

    Examples:
        >>> needleman_wunsch('dva', 'deeva')
        1.0
        >>> needleman_wunsch('dva', 'deeve', 0.0)
        2.0
        >>> needleman_wunsch('dva', 'deeve', 1.0, sim_score=lambda s1, s2 : (2.0 if s1 == s2 else -1.0))
        1.0
        >>> needleman_wunsch('GCATGCUA', 'GATTACA', gap_cost=0.5, sim_score=lambda s1, s2 : (1.0 if s1 == s2 else -1.0))
        2.5
    """
    dist_mat = np.zeros((len(string1) + 1, len(string2) + 1), dtype=np.float)
    for i in _range(len(string1) + 1):
        dist_mat[i, 0] = -(i * gap_cost)
    for j in _range(len(string2) + 1):
        dist_mat[0, j] = -(j * gap_cost)
    for i in _range(1, len(string1) + 1):
        for j in _range(1, len(string2) + 1):
            match = dist_mat[i - 1, j - 1] + sim_score(string1[i - 1], string2[j - 1])
            delete = dist_mat[i - 1, j] - gap_cost
            insert = dist_mat[i, j - 1] - gap_cost
            dist_mat[i, j] = max(match, delete, insert)
    return dist_mat[dist_mat.shape[0] - 1, dist_mat.shape[1] - 1]


@utils.sim_check_for_none
@utils.sim_check_for_string_inputs
def smith_waterman(string1, string2, gap_cost=1.0, sim_score=sim_ident):
    """
    Computes the Smith-Waterman measure between two strings.

    The Smith–Waterman algorithm performs local sequence alignment; that is, for determining similar regions
    between two strings. Instead of looking at the total sequence, the Smith–Waterman algorithm compares segments of
    all possible lengths and optimizes the similarity measure.


    Args:
        string1,string2 (str) : Input strings

        gap_cost (float) : Cost of gap (defaults to 1.0)

        sim_score (function) : Similarity function to give a score for the correspondence between characters. Defaults
            to an identity function, where if two characters are same it returns 1 else returns 0.

    Returns:
        Smith-Waterman measure (float)

    Raises:
        TypeError : If the inputs are not strings

    Examples:
        >>> smith_waterman('cat', 'hat')
        2.0
        >>> smith_waterman('dva', 'deeve', 2.2)
        1.0
        >>> smith_waterman('dva', 'deeve', 1, sim_score=lambda s1, s2 : (2 if s1 == s2 else -1))
        2.0
        >>> smith_waterman('GCATAGCU', 'GATTACA', gap_cost=1.4, sim_score=lambda s1, s2 : (1.5 if s1 == s2 else 0.5))
        6.5
    """
    dist_mat = np.zeros((len(string1) + 1, len(string2) + 1), dtype=np.float)
    max_value = 0
    for i in _range(1, len(string1) + 1):
        for j in _range(1, len(string2) + 1):
            match = dist_mat[i - 1, j - 1] + sim_score(string1[i - 1], string2[j - 1])
            delete = dist_mat[i - 1, j] - gap_cost
            insert = dist_mat[i, j - 1] - gap_cost
            dist_mat[i, j] = max(0, match, delete, insert)
            max_value = max(max_value, dist_mat[i, j])
    return max_value


# ---------------------- token based similarity measures  ----------------------

# ---------------------- set based similarity measures  ----------------------
@utils.sim_check_for_none
@utils.sim_check_for_list_or_set_inputs
@utils.sim_check_for_exact_match
@utils.sim_check_for_empty
def cosine(set1, set2):
    """
    Computes the cosine similarity between two sets.

    For two sets X and Y, the cosine similarity is:

    :math:`cosine(X, Y) = \\frac{|X \\cap Y|}{\\sqrt{|X| \\cdot |Y|}}`


    Args:
        set1,set2 (set or list): Input sets (or lists). Input lists are converted to sets.

    Returns:
        Cosine similarity (float)

    Raises:
        TypeError : If the inputs are not sets (or lists).

    Examples:
     >>> cosine(['data', 'science'], ['data'])
     0.7071067811865475
     >>> cosine(['data', 'data', 'science'], ['data', 'management'])
     0.4999999999999999
     >>> cosine([], ['data'])
     0.0

    References:
        * String similarity joins: An Experimental Evaluation (VLDB 2014)
        * Project flamingo : Mike carey, Vernica
    """

    if not isinstance(set1, set):
        set1 = set(set1)
    if not isinstance(set2, set):
        set2 = set(set2)
    return float(len(set1 & set2)) / (math.sqrt(float(len(set1))) * math.sqrt(float(len(set2))))


@utils.sim_check_for_none
@utils.sim_check_for_list_or_set_inputs
@utils.sim_check_for_exact_match
@utils.sim_check_for_empty
def jaccard(set1, set2):
    """
    Computes the Jaccard measure between two sets.

    The Jaccard measure, also known as the Jaccard similarity coefficient, is a statistic used for comparing
    the similarity and diversity of sample sets. The Jaccard coefficient measures similarity between finite sample
    sets, and is defined as the size of the intersection divided by the size of the union of the sample sets.


    For two sets X and Y, the Jaccard measure is:

    :math:`jaccard(X, Y) = \\frac{|X \\cap Y|}{|X| \\cup |Y|}`


    Args:
        set1,set2 (set or list): Input sets (or lists). Input lists are converted to sets.

    Returns:
        Jaccard similarity (float)

    Raises:
        TypeError : If the inputs are not sets (or lists).

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
@utils.sim_check_for_list_or_set_inputs
@utils.sim_check_for_exact_match
@utils.sim_check_for_empty
def overlap_coefficient(set1, set2):
    """
    Computes the overlap coefficient between two sets.

    The overlap coefficient is a similarity measure related to the Jaccard
    measure  that measures the overlap between two sets, and is defined as the size of the intersection divided by
    the smaller of the size of the two sets.

    For two sets X and Y, the overlap coefficient is:

    :math:`overlap\\_coefficient(X, Y) = \\frac{|X \\cap Y|}{\\min(|X|, |Y|)}`

    Args:
        set1,set2 (set or list): Input sets (or lists). Input lists are converted to sets.

    Returns:
        Overlap coefficient (float)

    Raises:
        TypeError : If the inputs are not sets (or lists).

    Examples:
        >>> (overlap_coefficient([], [])
        1.0
        >>> overlap_coefficient([], ['data'])
        0
        >>> overlap_coefficient(['data', 'science'], ['data'])
        1.0

    References:
        * Wikipedia article : https://en.wikipedia.org/wiki/Overlap_coefficient
        * Simmetrics library

    """

    if not isinstance(set1, set):
        set1 = set(set1)
    if not isinstance(set2, set):
        set2 = set(set2)

    return float(len(set1 & set2)) / min(len(set1), len(set2))


# ---------------------- bag based similarity measures  ----------------------


# hybrid similarity measures
@utils.sim_check_for_none
@utils.sim_check_for_list_or_set_inputs
@utils.sim_check_for_exact_match
@utils.sim_check_for_empty
def monge_elkan(bag1, bag2, sim_func=levenshtein):
    """
    Compute Monge-Elkan similarity measure between two bags (lists).

    The Monge-Elkan similarity measure is a type of Hybrid similarity measure that combine the benefits of
    sequence-based and set-based methods. This can be effective for domains in which more control is needed
    over the similarity measure. It implicitly uses a secondary similarity measure, such as levenshtein to compute
    over all similarity score.

    Args:
        bag1,bag2 (list): Input lists

        sim_func (function): Secondary similarity function. This is expected to be a sequence-based
            similarity measure (defaults to levenshtein)

    Returns:
        Monge-Elkan similarity score (float)

    Raises:
        TypeError : If the inputs are not lists


    Examples:
        >>> monge_elkan(['Niall'], ['Neal'])
        2.0
        >>> monge_elkan([''], ['a'])
        1.0
        >>> monge_elkan(['Niall'], ['Nigel'])
        2.0

    References:
        * Principles of Data Integration book
    """
    sum_of_maxes = 0
    for t1 in bag1:
        max_sim = float('-inf')
        for t2 in bag2:
            max_sim = max(max_sim, sim_func(t1, t2))
        sum_of_maxes += max_sim
    sim = float(sum_of_maxes) / float(len(bag1))
    return sim

