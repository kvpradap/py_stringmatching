from __future__ import unicode_literals
import unittest

from py_stringmatching.tokenizers import qgram

class QgramTestCases(unittest.TestCase):
    def test_qgrams(self):
        self.assertEqual(qgram(''), [])
        self.assertEqual(qgram('a'), [])
        self.assertEqual(qgram('aa'), ['aa'])
        self.assertEqual(qgram('database'), ['da','at','ta','ab','ba','as','se'])
        self.assertEqual(qgram('d', 1), ['d'])


