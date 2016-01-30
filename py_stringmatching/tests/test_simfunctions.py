from __future__ import unicode_literals
from nose.tools import *
import unittest

from py_stringmatching.simfunctions import levenshtein

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
