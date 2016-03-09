# coding=utf-8
from __future__ import division
from __future__ import unicode_literals

import Levenshtein
import math, collections

from py_stringmatching import utils
import numpy as np
from .compat import _range


def sim_ident(s1, s2):
    return int(s1 == s2)

# ---------------------- sequence based similarity measures  ----------------------


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
        TypeError : If the inputs are not strings or if one of the inputs is None.

    Examples:
        >>> affine('dva', 'deeva')
        1.5
        >>> affine('dva', 'deeve', gap_start=2, gap_continuation=0.5)
        -0.5
        >>> affine('AAAGAATTCA', 'AAATCA', gap_continuation=0.2, sim_score=lambda s1, s2 : (int(1 if s1 == s2 else 0)))
        4.4
    """
    utils.sim_check_for_none(string1, string2)
    utils.tok_check_for_string_input(string1, string2)
    if utils.sim_check_for_empty(string1, string2):
        return 0

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
        TypeError : If the inputs are not strings or if one of the inputs is None.


    Examples:
        >>> jaro('MARTHA', 'MARHTA')
        0.9444444444444445
        >>> jaro('DWAYNE', 'DUANE')
        0.8222222222222223
        >>> jaro('DIXON', 'DICKSONX')
        0.7666666666666666


    """
    utils.sim_check_for_none(string1, string2)
    utils.tok_check_for_string_input(string1, string2)
    if utils.sim_check_for_empty(string1, string2):
        return 0

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
        TypeError : If the inputs are not strings or if one of the inputs is None.


    Examples:
        >>> jaro_winkler('MARTHA', 'MARHTA')
        0.9611111111111111
        >>> jaro_winkler('DWAYNE', 'DUANE')
        0.84
        >>> jaro_winkler('DIXON', 'DICKSONX')
        0.8133333333333332

    """
    utils.sim_check_for_none(string1, string2)
    utils.tok_check_for_string_input(string1, string2)
    if utils.sim_check_for_empty(string1, string2):
        return 0

    jw_score = jaro(string1, string2)
    min_len = min(len(string1), len(string2))
    j = min(min_len, 4)
    i = 0
    while i < j and string1[i] == string2[i] and string1[i]:
        i += 1
    if i:
        jw_score += i * prefix_weight * (1 - jw_score)

    return jw_score


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
        TypeError : If the inputs are not strings or if one of the inputs is None.
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
    utils.sim_check_for_none(string1, string2)
    utils.tok_check_for_string_input(string1, string2)
    utils.sim_check_for_same_len(string1, string2)
    return sum(bool(ord(c1) - ord(c2)) for c1, c2 in zip(string1, string2))


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
    utils.sim_check_for_none(string1, string2)
    utils.sim_check_for_string_inputs(string1, string2)

    return Levenshtein.distance(string1, string2)


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
        TypeError : If the inputs are not strings or if one of the inputs is None.

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
    utils.sim_check_for_none(string1, string2)
    utils.sim_check_for_string_inputs(string1, string2)

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
        TypeError : If the inputs are not strings or if one of the inputs is None.

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
    utils.sim_check_for_none(string1, string2)
    utils.sim_check_for_string_inputs(string1, string2)

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
        TypeError : If the inputs are not sets (or lists) or if one of the inputs is None.

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
    utils.sim_check_for_none(set1, set2)
    utils.sim_check_for_list_or_set_inputs(set1, set2)
    if utils.sim_check_for_exact_match(set1, set2):
        return 1.0
    if utils.sim_check_for_empty(set1, set2):
        return 0
    if not isinstance(set1, set):
        set1 = set(set1)
    if not isinstance(set2, set):
        set2 = set(set2)
    return float(len(set1 & set2)) / (math.sqrt(float(len(set1))) * math.sqrt(float(len(set2))))


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
        TypeError : If the inputs are not sets (or lists) or if one of the inputs is None.

    Examples:
        >>> jaccard(['data', 'science'], ['data'])
        0.5
        >>> jaccard({1, 1, 2, 3, 4}, {2, 3, 4, 5, 6, 7, 7, 8})
        0.375
        >>> jaccard(['data', 'management'], ['data', 'data', 'science'])
        0.3333333333333333
    """
    utils.sim_check_for_none(set1, set2)
    utils.sim_check_for_list_or_set_inputs(set1, set2)
    if utils.sim_check_for_exact_match(set1, set2):
        return 1.0
    if utils.sim_check_for_empty(set1, set2):
        return 0
    if not isinstance(set1, set):
        set1 = set(set1)
    if not isinstance(set2, set):
        set2 = set(set2)
    return float(len(set1 & set2)) / float(len(set1 | set2))


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
        TypeError : If the inputs are not sets (or lists) or if one of the inputs is None.

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
    utils.sim_check_for_none(set1, set2)
    utils.sim_check_for_list_or_set_inputs(set1, set2)
    if utils.sim_check_for_exact_match(set1, set2):
        return 1.0
    if utils.sim_check_for_empty(set1, set2):
        return 0
    if not isinstance(set1, set):
        set1 = set(set1)
    if not isinstance(set2, set):
        set2 = set(set2)

    return float(len(set1 & set2)) / min(len(set1), len(set2))


# ---------------------- bag based similarity measures  ----------------------
def tfidf(bag1, bag2, corpus_list = None, dampen=False):
    """
    Compute tfidf measures between two lists given the corpus information.
    This measure employs the notion of TF/IDF score commonly used in information retrieval (IR) to find documents that
    are relevant to keyword queries.
    The intuition underlying the TF/IDF measure is that two strings are similar if they share distinguishing terms.

    Args:
        bag1,bag2 (list): Input lists

        corpus_list (list of lists): Corpus list (default is set to None) of strings. If set to None,
            the input list are considered the only corpus.

        dampen (boolean): Flag to indicate whether 'log' should be applied to tf and idf measure.

    Returns:
        TF-IDF measure between the input lists (float)

    Raises:
        TypeError : If the inputs are not lists or if one of the inputs is None


    Examples:
        >>> tfidf(['a', 'b', 'a'], ['a', 'c'], [['a', 'b', 'a'], ['a', 'c'], ['a']])
        0.17541160386140586
        >>> tfidf(['a', 'b', 'a'], ['a', 'c'], [['a', 'b', 'a'], ['a', 'c'], ['a'], ['b']], True)
        0.11166746710505392
        >>> tfidf(['a', 'b', 'a'], ['a'], [['a', 'b', 'a'], ['a', 'c'], ['a']])
        0.5547001962252291
        >>> tfidf(['a', 'b', 'a'], ['a'], [['x', 'y'], ['w'], ['q']])
        0.0
        >>> tfidf(['a', 'b', 'a'], ['a'], [['x', 'y'], ['w'], ['q']], True)
        0.0
        >>> tfidf(['a', 'b', 'a'], ['a'])
        0.7071067811865475
    """
    utils.sim_check_for_none(bag1, bag2)
    utils.sim_check_for_list_or_set_inputs(bag1, bag2)
    if utils.sim_check_for_exact_match(bag1, bag2):
        return 1.0
    if utils.sim_check_for_empty(bag1, bag2):
        return 0
    if corpus_list is None:
        corpus_list = [bag1, bag2]
    corpus_size = len(corpus_list)
    tf_x, tf_y = collections.Counter(bag1), collections.Counter(bag2)
    element_freq = {}
    total_unique_elements = set()
    for document in corpus_list:
        temp_set = set()
        for element in document:
            if element in bag1 or element in bag2:
                temp_set.add(element)
                total_unique_elements.add(element)
        for element in temp_set:
            element_freq[element] = element_freq[element]+1 if element in element_freq else 1
    idf_element, v_x, v_y, v_x_y, v_x_2, v_y_2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for element in total_unique_elements:
        idf_element = corpus_size * 1.0 / element_freq[element]
        v_x = 0 if element not in tf_x else (math.log(idf_element) * math.log(tf_x[element] + 1)) if dampen else (idf_element * tf_x[element])
        v_y = 0 if element not in tf_y else (math.log(idf_element) * math.log(tf_y[element] + 1)) if dampen else (idf_element * tf_y[element])
        v_x_y += v_x * v_y
        v_x_2 += v_x * v_x
        v_y_2 += v_y * v_y
    return 0.0 if v_x_y == 0 else v_x_y/(math.sqrt(v_x_2) * math.sqrt(v_y_2))


# hybrid similarity measures
def monge_elkan(bag1, bag2, sim_func=jaro_winkler):
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
        TypeError : If the inputs are not lists or if one of the inputs is None


    Examples:
        >>> monge_elkan(['Niall'], ['Neal'])
        0.8049999999999999
        >>> monge_elkan(['Comput.', 'Sci.', 'and', 'Eng.', 'Dept.,', 'University', 'of', 'California,', 'San', 'Diego'], ['Department', 'of', 'Computer', 'Science,', 'Univ.', 'Calif.,', 'San', 'Diego'])
        0.8677218614718616
        >>> monge_elkan(['Comput.', 'Sci.', 'and', 'Eng.', 'Dept.,', 'University', 'of', 'California,', 'San', 'Diego'], ['Department', 'of', 'Computer', 'Science,', 'Univ.', 'Calif.,', 'San', 'Diego'], sim_func=needleman_wunsch)
        2.0
        >>> monge_elkan(['Comput.', 'Sci.', 'and', 'Eng.', 'Dept.,', 'University', 'of', 'California,', 'San', 'Diego'], ['Department', 'of', 'Computer', 'Science,', 'Univ.', 'Calif.,', 'San', 'Diego'], sim_func=affine)
        2.25
        >>> monge_elkan([''], ['a'])
        0.0
        >>> monge_elkan(['Niall'], ['Nigel'])
        0.7866666666666667

    References:
        * Principles of Data Integration book
    """
    utils.sim_check_for_none(bag1, bag2)
    utils.sim_check_for_list_or_set_inputs(bag1, bag2)
    if utils.sim_check_for_exact_match(bag1, bag2):
        return 1.0
    if utils.sim_check_for_empty(bag1, bag2):
        return 0
    sum_of_maxes = 0
    for t1 in bag1:
        max_sim = float('-inf')
        for t2 in bag2:
            max_sim = max(max_sim, sim_func(t1, t2))
        sum_of_maxes += max_sim
    sim = float(sum_of_maxes) / float(len(bag1))
    return sim


def soft_tfidf(bag1, bag2, corpus_list=None, sim_func=jaro, threshold=0.5):
    """
    Compute Soft-tfidf measures between two lists given the corpus information.

    Args:
        bag1,bag2 (list): Input lists

        corpus_list (list of lists): Corpus list (default is set to None) of strings. If set to None,
            the input list are considered the only corpus

        sim_func (func): Secondary similarity function. This should return a similarity score between two strings (optional),
            default is jaro similarity measure

        threshold (float): Threshold value for the secondary similarity function (defaults to 0.5). If the similarity
            of a token pair exceeds the threshold, then the token pair is considered a match.

    Returns:
        Soft TF-IDF measure between the input lists

    Raises:
        TypeError : If the inputs are not lists or if one of the inputs is None.

    Examples:
        >>> soft_tfidf(['a', 'b', 'a'], ['a', 'c'], [['a', 'b', 'a'], ['a', 'c'], ['a']], sim_func=jaro, threshold=0.8)
        0.17541160386140586
        >>> soft_tfidf(['a', 'b', 'a'], ['a'], [['a', 'b', 'a'], ['a', 'c'], ['a']], threshold=0.9)
        0.5547001962252291
        >>> soft_tfidf(['a', 'b', 'a'], ['a'], [['x', 'y'], ['w'], ['q']])
        0.0
        >>> soft_tfidf(['aa', 'bb', 'a'], ['ab', 'ba'], sim_func=affine, threshold=0.6)
        0.81649658092772592

    References:
        * Principles of Data Integration book
    """
    utils.sim_check_for_none(bag1, bag2)
    utils.sim_check_for_list_or_set_inputs(bag1, bag2)
    if utils.sim_check_for_exact_match(bag1, bag2):
        return 1.0
    if utils.sim_check_for_empty(bag1, bag2):
        return 0
    if corpus_list is None:
        corpus_list = [bag1, bag2]
    corpus_size = len(corpus_list) * 1.0
    tf_x, tf_y = collections.Counter(bag1), collections.Counter(bag2)
    element_freq = {}
    total_unique_elements = set()
    for document in corpus_list:
        temp_set = set()
        for element in document:
            if element in bag1 or element in bag2:
                temp_set.add(element)
                total_unique_elements.add(element)
        for element in temp_set:
            element_freq[element] = element_freq[element]+1 if element in element_freq else 1
    similarity_map = {}
    for x in bag1:
        if x not in similarity_map:
            max_score = 0.0
            for y in bag2:
                score = sim_func(x,y)
                if score > threshold and score > max_score:
                    similarity_map[x] = utils.Similarity(x, y, score)
                    max_score = score
    result, v_x_2, v_y_2 = 0.0, 0.0, 0.0
    for element in total_unique_elements:
        # numerator
        if element in similarity_map:
            sim = similarity_map[element]
            idf_first = corpus_size if sim.first_string not in element_freq else corpus_size / \
                                                                                 element_freq[sim.first_string]
            idf_second = corpus_size if sim.second_string not in element_freq else corpus_size / \
                                                                                   element_freq[sim.second_string]
            v_x = 0 if sim.first_string not in tf_x else idf_first * tf_x[sim.first_string]
            v_y = 0 if sim.second_string not in tf_y else idf_second * tf_y[sim.second_string]
            result += v_x * v_y * sim.similarity_score
        # denominator
        idf = corpus_size if element not in element_freq else corpus_size / element_freq[element]
        v_x = 0 if element not in tf_x else idf * tf_x[element]
        v_x_2 += v_x * v_x
        v_y = 0 if element not in tf_y else idf * tf_y[element]
        v_y_2 += v_y * v_y
    return result if v_x_2 == 0 else result/(math.sqrt(v_x_2) * math.sqrt(v_y_2))