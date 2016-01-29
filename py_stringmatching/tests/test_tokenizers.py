from __future__ import unicode_literals
import unittest

from py_stringmatching.tokenizers import qgram

class QgramTestCases():
    def test_qgrams(self):
        self.assertEqual(sorted(qgram('')), [])
    pass