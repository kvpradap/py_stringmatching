from __future__ import unicode_literals

import unittest
from nose.tools import *

from py_stringmatching.tokenizers import qgram, delimiter, whitespace


class QgramTestCases(unittest.TestCase):
    def test_qgrams_valid(self):
        self.assertEqual(qgram(''), [])
        self.assertEqual(qgram('a'), [])
        self.assertEqual(qgram('aa'), ['aa'])
        self.assertEqual(qgram('database'), ['da', 'at', 'ta', 'ab', 'ba', 'as', 'se'])
        self.assertEqual(qgram('d', 1), ['d'])
        self.assertEqual(qgram('database', 3), ['dat', 'ata', 'tab', 'aba', 'bas', 'ase'])

    @raises(TypeError)
    def test_qgrams_none(self):
        self.assertEqual(qgram(None), [])


class DelimiterTestCases(unittest.TestCase):
    def test_delimiter_valid(self):
        self.assertEqual(delimiter('data science'), ['data', 'science'])
        self.assertEqual(delimiter('data,science', ','), ['data', 'science'])
        self.assertEqual(delimiter('data science', ','), ['data science'])
        self.assertEqual(delimiter('data$#$science', '$#$'), ['data', 'science'])

    def test_delimiter_invalid1(self):
        self.assertEqual(delimiter('data science', None), ['data', 'science'])

    @raises(TypeError)
    def test_delimiter_invalid2(self):
        self.assertEqual(delimiter('data science', 10), ['data', 'science'])

    @raises(TypeError)
    def test_delimiter_invalid3(self):
        self.assertEqual(delimiter(None), [])


class WhiteSpaceTestCases(unittest.TestCase):
    def test_delimiter_valid(self):
        self.assertEqual(whitespace('data science'), ['data', 'science'])
        self.assertEqual(whitespace('data        science'), ['data', 'science'])
        self.assertEqual(whitespace('data   science'), ['data', 'science'])
        self.assertEqual(whitespace('data\tscience'), ['data', 'science'])
        self.assertEqual(whitespace('data\nscience'), ['data', 'science'])

    @raises(TypeError)
    def test_delimiter_invalid(self):
        self.assertEqual(whitespace(None))
