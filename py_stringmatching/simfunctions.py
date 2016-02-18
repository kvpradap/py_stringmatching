# coding=utf-8
from __future__ import division
from __future__ import unicode_literals

import Levenshtein
import math, collections

from py_stringmatching import utils
from py_stringmatching.utils import Similarity
import numpy as np
from .compat import _range


def sim_ident(s1, s2):
    return int(s1 == s2)

# @todo: add examples in the comments
# ---------------------- sequence based similarity measures  ----------------------


@utils.sim_check_for_none
@utils.tok_check_for_string_input
@utils.sim_check_for_empty
def affine(string1, string2, gap_start=-1, gap_continuation=-0.5, sim_score=sim_ident):
    """
    Calculates the Affine gap measure similarity score between two strings.
    This is calculated according to the description provided in the Data Integration book.

    Args:
        string1,string2 (str) : Input strings

        gap_start (float): Cost for the gap at the start (default value is -1)

        gap_continuation (float) : Cost for the gap continuation (default value is -0.5)

        sim_score (function) : Function computing similarity score between two chars (rep as strings)
        (default value is identity)

    Returns:
        If string1 and string2 are valid strings then
        Affine gap measure (float) between two strings is returned.

    Examples:
        >>> affine('dva', 'deeva')
        1.5
        >>> affine('dva', 'deeve', gap_start=-2, gap_continuation=-0.5)
        -0.5
        >>> affine('AAAGAATTCA', 'AAATCA', gap_continuation=-0.2, sim_score=lambda s1, s2 : (int(1 if s1 == s2 else 0)))
        4.4
    """
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
@utils.tok_check_for_string_input
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


@utils.sim_check_for_none
@utils.tok_check_for_string_input
@utils.sim_check_for_same_len
def hamming_distance(string1, string2):
    """
    This function calculates the hamming distance between the two equal length strings.
    It is the number of positions at which the corresponding symbols are different.

    Args:
        string1, string2 (str): Input strings

    Returns:
        If string1 and string2 are of same length the
        Hamming Distance distance (int) between two strings is returned.

    Notes:
        This function internally uses python-levenshtein package

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
    # return Levenshtein.hamming(string1, string2)
    return sum(bool(ord(c1) - ord(c2)) for c1, c2 in zip(string1, string2))

@utils.sim_check_for_none
@utils.sim_check_for_string_inputs
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


@utils.sim_check_for_none
@utils.sim_check_for_string_inputs
def needleman_wunsch(string1, string2, gap_cost=1, sim_score=sim_ident):
    """
    Calculates the Needleman-Wunsch similarity score between two strings.

    Args:
        string1, string2 (str) : Input strings

        gap_cost (int) : Cost of gap (default value is 1)

        sim_score (function) : Similarity function for two chars (rep as strings)
        (default value is identity, i.e for the same character the score is 1, else
        the score is 0)

    Returns:
        If string1 and string2 are valid strings then
        Needleman-Wunsch similarity (int) between two strings is returned.

    Examples:
        >>> needleman_wunsch('dva', 'deeva')
        0
        >>> needleman_wunsch('dva', 'deeve', 0)
        2
        >>> needleman_wunsch('dva', 'deeve', 1, sim_score=lambda s1, s2 : (int(2 if s1 == s2 else -1)))
        1
        >>> needleman_wunsch('GCATGCU', 'GATTACA', gap_cost=1, sim_score=lambda s1, s2 : (int(1 if s1 == s2 else -1)))
        0
    """
    dist_mat = np.zeros((len(string1) + 1, len(string2) + 1), dtype=np.int)
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
def smith_waterman(string1, string2, gap_cost=1, sim_score=sim_ident):
    """
    Calculates the Smith-Waterman similarity score between two strings. Cf. https://en.wikipedia.org/wiki/Smith–Waterman_algorithm, https://github.com/Simmetrics

    Args:
        string1, string2 (str) : Input strings

        gap_cost (int) : Cost of gap (default value is 1)

        sim_score (function) : Similarity function for two chars (rep as strings)
        (default value is identity, i.e for the same character the score is 1, else
        the score is 0)

    Returns:
        If string1 and string2 are valid strings then
        Needleman-Wunsch similarity (int) between two strings is returned.

    Examples:
        >>> smith_waterman('cat', 'hat')
        2
        >>> smith_waterman('dva', 'deeve', 0)
        2
        >>> smith_waterman('dva', 'deeve', 1, sim_score=lambda s1, s2 : (int(2 if s1 == s2 else -1)))
        2
        >>> smith_waterman('GCATAGCU', 'GATTACA', gap_cost=1, sim_score=lambda s1, s2 : (int(1 if s1 == s2 else 0)))
        3
    """
    dist_mat = np.zeros((len(string1) + 1, len(string2) + 1), dtype=np.int)
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
    Cosine similarity between two sets.

    Args:
        set1, set2 (list): input sets

    Returns:
        If set1 and set2 are valid sets then
            cosine similarity (float) between two bags is returned.

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
@utils.sim_check_for_list_or_set_inputs
@utils.sim_check_for_exact_match
@utils.sim_check_for_empty
def overlap_coefficient(set1, set2):
    """

    Args:
        set1, set2 (set): Input sets.


    Returns:
        If set1 and set2 are valid sets/lists or single values then
            overlap similarity (float) between two sets is returned.
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
    sim = float(sum_of_maxes) / float(len(bag1))
    return sim


@utils.sim_check_for_none
@utils.sim_check_for_list_or_set_inputs
@utils.sim_check_for_exact_match
@utils.sim_check_for_empty
def soft_tfidf(x_tok, y_tok, corpus_list=None, sim_func=jaro, threshold=0.5):
    """
    Compute Soft-tfidf measures between two lists given the corpus information.
    This measure is similar in spirit to the generalized Jaccard measure, except that it uses the TF/IDF measure
    instead of Jaccard measure as the higher level similarity measure.

    Args:
        x_tok, y_tok (list): Input lists
        corpus_list (list of lists): Corpus list (default is set to None). If set to None,
        the input list are considered the only corpus
        sim_func (func(string1, string2)): this should return a similarity score between two strings (optional), default is jaro similarity measure
        threshold (float): threshold value for the given similarity function. Defaults to 0.5


    Returns:
        Soft TF-IDF measure between the input lists

    Examples:
        >>> soft_tfidf(['a', 'b', 'a'], ['a', 'c'], [['a', 'b', 'a'], ['a', 'c'], ['a']], sim_func=jaro, threshold=0.8)
        0.17541160386140586
        >>> soft_tfidf(['a', 'b', 'a'], ['a'], [['a', 'b', 'a'], ['a', 'c'], ['a']], threshold=0.9)
        0.5547001962252291
        >>> soft_tfidf(['a', 'b', 'a'], ['a'], [['x', 'y'], ['w'], ['q']])
        0.0
        >>> soft_tfidf(['aa', 'bb', 'a'], ['ab', 'ba'], sim_func=affine, threshold=0.6)
        0.81649658092772592
    """
    if corpus_list is None:
        corpus_list = [x_tok, y_tok]
    corpus_size = len(corpus_list) * 1.0
    tf_x, tf_y = collections.Counter(x_tok), collections.Counter(y_tok)
    element_freq = {}
    total_unique_elements = set()
    for document in corpus_list:
        temp_set = set()
        for element in document:
            if element in x_tok or element in y_tok:
                temp_set.add(element)
                total_unique_elements.add(element)
        for element in temp_set:
            element_freq[element] = element_freq[element]+1 if element in element_freq else 1
    similarity_map = {}
    for x in x_tok:
        if x not in similarity_map:
            max_score = 0.0
            for y in y_tok:
                score = sim_func(x,y)
                if score > threshold and score > max_score:
                    similarity_map[x] = Similarity(x, y, score)
                    max_score = score
    result, v_x_2, v_y_2 = 0.0, 0.0, 0.0
    for element in total_unique_elements:
        # numerator
        if element in similarity_map:
            sim = similarity_map[element]
            idf_first = corpus_size if sim.first_string not in element_freq else corpus_size / element_freq[sim.first_string]
            idf_second = corpus_size if sim.second_string not in element_freq else corpus_size / element_freq[sim.second_string]
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