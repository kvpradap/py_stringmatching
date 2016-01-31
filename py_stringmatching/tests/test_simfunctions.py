from __future__ import unicode_literals
import math
from nose.tools import *
import unittest

from py_stringmatching.simfunctions import levenshtein, jaro, jaro_winkler
from py_stringmatching.simfunctions import overlap
from py_stringmatching.simfunctions import cosine

from py_stringmatching.tokenizers import qgram, whitespace

# ---------------------- sequence based similarity measures  ----------------------
class JaroTestCases(unittest.TestCase):
    def test_valid_input(self):
        # https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance
        self.assertAlmostEqual(jaro('MARTHA', 'MARHTA'), 0.9444444444444445)
        self.assertAlmostEqual(jaro('DWAYNE', 'DUANE'), 0.8222222222222223)
        self.assertAlmostEqual(jaro('DIXON', 'DICKSONX'), 0.7666666666666666)

    @raises(TypeError)
    def test_invalid_input1(self):
        jaro(None, 'MARHTA')

    @raises(TypeError)
    def test_invalid_input2(self):
        jaro('MARHTA', None)

    @raises(TypeError)
    def test_invalid_input2(self):
        jaro(None, None)


class JaroWinklerTestCases(unittest.TestCase):
    def test_valid_input(self):
        # https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance
        self.assertAlmostEqual(jaro_winkler('MARTHA', 'MARHTA'), 0.9611111111111111)
        self.assertAlmostEqual(jaro_winkler('DWAYNE', 'DUANE'), 0.84)
        self.assertAlmostEqual(jaro_winkler('DIXON', 'DICKSONX'), 0.8133333333333332)

    @raises(TypeError)
    def test_invalid_input1(self):
        jaro_winkler(None, 'MARHTA')

    @raises(TypeError)
    def test_invalid_input2(self):
        jaro_winkler('MARHTA', None)

    @raises(TypeError)
    def test_invalid_input2(self):
        jaro_winkler(None, None)




class LevenshteinTestCases(unittest.TestCase):
    def test_valid_input(self):
        # http://oldfashionedsoftware.com/tag/levenshtein-distance/
        self.assertEqual(levenshtein('a', ''), 1)
        self.assertEqual(levenshtein('', 'a'), 1)
        self.assertEqual(levenshtein('abc', ''), 3)
        self.assertEqual(levenshtein('', 'abc'), 3)
        self.assertEqual(levenshtein('', ''), 0)
        self.assertEqual(levenshtein('a', 'a'), 0)
        self.assertEqual(levenshtein('abc', 'abc'), 0)
        self.assertEqual(levenshtein('', 'a'), 1)
        self.assertEqual(levenshtein('a', 'ab'), 1)
        self.assertEqual(levenshtein('b', 'ab'), 1)
        self.assertEqual(levenshtein('ac', 'abc'), 1)
        self.assertEqual(levenshtein('abcdefg', 'xabxcdxxefxgx'), 6)
        self.assertEqual(levenshtein('a', ''), 1)
        self.assertEqual(levenshtein('ab', 'a'), 1)
        self.assertEqual(levenshtein('ab', 'b'), 1)
        self.assertEqual(levenshtein('abc', 'ac'), 1)
        self.assertEqual(levenshtein('xabxcdxxefxgx', 'abcdefg'), 6)
        self.assertEqual(levenshtein('a', 'b'), 1)
        self.assertEqual(levenshtein('ab', 'ac'), 1)
        self.assertEqual(levenshtein('ac', 'bc'), 1)
        self.assertEqual(levenshtein('abc', 'axc'), 1)
        self.assertEqual(levenshtein('xabxcdxxefxgx', '1ab2cd34ef5g6'), 6)
        self.assertEqual(levenshtein('example', 'samples'), 3)
        self.assertEqual(levenshtein('sturgeon', 'urgently'), 6)
        self.assertEqual(levenshtein('levenshtein', 'frankenstein'), 6)
        self.assertEqual(levenshtein('distance', 'difference'), 5)
        self.assertEqual(levenshtein('java was neat', 'scala is great'), 7)
    @raises(TypeError)
    def test_invalid_input1(self):
        levenshtein('a', None)
    @raises(TypeError)
    def test_invalid_input2(self):
        levenshtein(None, 'b')
    @raises(TypeError)
    def test_invalid_input3(self):
        levenshtein(None, None)


# ---------------------- token based similarity measures  ----------------------

# ---------------------- set based similarity measures  ----------------------
class OverlapTestCases(unittest.TestCase):
    def test_valid_input(self):
        self.assertEqual(overlap([], []), 1.0)
        self.assertEqual(overlap(['data',  'science'], ['data']), 1.0/min(2.0, 1.0))
        self.assertEqual(overlap(['data', 'science'], ['science', 'good']), 1.0/min(2.0, 3.0))
        self.assertEqual(overlap([], ['data']), 0)
        self.assertEqual(overlap(['data', 'data', 'science'],['data', 'management']), 1.0/min(3.0, 2.0))

    @raises(TypeError)
    def test_invalid_input1(self):
        overlap(['a'], None)
    @raises(TypeError)
    def test_invalid_input2(self):
        overlap(None, ['b'])
    @raises(TypeError)
    def test_invalid_input3(self):
        overlap(None, None)


# ---------------------- bag based similarity measures  ----------------------
class CosineTestCases(unittest.TestCase):
    def test_valid_input(self):
        NONQ_FROM = 'The quick brown fox jumped over the lazy dog.'
        NONQ_TO = 'That brown dog jumped over the fox.'
        self.assertEqual(cosine([], []), 1) # check-- done. both simmetrics, abydos return 1.
        self.assertEqual(cosine(['the', 'quick'], []), 0)
        self.assertEqual(cosine([], ['the', 'quick']), 0)
        self.assertAlmostEqual(cosine(whitespace(NONQ_TO), whitespace(NONQ_FROM)),
                               4/math.sqrt(9*7))

    @raises(TypeError)
    def test_invalid_input1(self):
        cosine(['a'], None)
    @raises(TypeError)
    def test_invalid_input2(self):
        cosine(None, ['b'])
    @raises(TypeError)
    def test_invalid_input3(self):
        cosine(None, None)
