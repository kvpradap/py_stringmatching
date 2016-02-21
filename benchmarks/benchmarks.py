# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

from py_stringmatching import simfunctions
from . import _short_string_1, _long_string_1, _medium_string_1, _short_string_2, _long_string_2, _medium_string_2
from . import _small_num_tokens_wi_rep, _small_num_tokens_wo_rep, _med_num_tokens_wi_rep, _med_num_tokens_wo_rep, \
    _large_num_tokens_wi_rep, _large_num_tokens_wo_rep
import Levenshtein


class TimeJaro:
    def time_short_short(self):
        simfunctions.jaro(_short_string_1, _short_string_2)

    def time_medium_medium(self):
        simfunctions.jaro(_medium_string_1, _medium_string_2)

    def time_long_long(self):
        simfunctions.jaro(_long_string_1, _long_string_2)

    def time_short_medium(self):
        simfunctions.jaro(_short_string_1, _medium_string_1)

    def time_short_long(self):
        simfunctions.jaro(_short_string_1, _long_string_1)

    def time_medium_long(self):
        simfunctions.jaro(_medium_string_1, _long_string_1)

class TimeJaro1:
    def time_short_short(self):
        Levenshtein.jaro(_short_string_1, _short_string_2)

    def time_medium_medium(self):
        Levenshtein.jaro(_medium_string_1, _medium_string_2)

    def time_long_long(self):
        Levenshtein.jaro(_long_string_1, _long_string_2)

    def time_short_medium(self):
        Levenshtein.jaro(_short_string_1, _medium_string_1)

    def time_short_long(self):
        Levenshtein.jaro(_short_string_1, _long_string_1)

    def time_medium_long(self):
        Levenshtein.jaro(_medium_string_1, _long_string_1)

# class TimeJaroWinkler:
#     def time_short_short(self):
#         simfunctions.jaro_winkler(_short_string_1, _short_string_2)
#
#     def time_medium_medium(self):
#         simfunctions.jaro_winkler(_medium_string_1, _medium_string_2)
#
#     def time_long_long(self):
#         simfunctions.jaro_winkler(_long_string_1, _long_string_2)
#
#     def time_short_medium(self):
#         simfunctions.jaro_winkler(_short_string_1, _medium_string_1)
#
#     def time_short_long(self):
#         simfunctions.jaro_winkler(_short_string_1, _long_string_1)
#
#     def time_medium_long(self):
#         simfunctions.jaro_winkler(_medium_string_1, _long_string_1)
#
#
#
# class TimeLevenshtein:
#     def time_short_short(self):
#         simfunctions.levenshtein(_short_string_1, _short_string_2)
#
#     def time_medium_medium(self):
#         simfunctions.levenshtein(_medium_string_1, _medium_string_2)
#
#     def time_long_long(self):
#         simfunctions.levenshtein(_long_string_1, _long_string_2)
#
#     def time_short_medium(self):
#         simfunctions.levenshtein(_short_string_1, _medium_string_1)
#
#     def time_short_long(self):
#         simfunctions.levenshtein(_short_string_1, _long_string_1)
#
#     def time_medium_long(self):
#         simfunctions.levenshtein(_medium_string_1, _long_string_1)
#
#
#
# class TimeOverlap:
#     def time_small_small_wo_rep(self):
#         simfunctions.overlap_coefficient(_small_num_tokens_wo_rep, _small_num_tokens_wo_rep)
#
#
#     def time_small_small_wi_rep(self):
#         simfunctions.overlap_coefficient(_small_num_tokens_wi_rep, _small_num_tokens_wi_rep)
#
#     def time_medium_medium_wo_rep(self):
#         simfunctions.overlap_coefficient(_med_num_tokens_wo_rep, _med_num_tokens_wo_rep)
#
#     def time_medium_medium_wi_rep(self):
#         simfunctions.overlap_coefficient(_med_num_tokens_wi_rep, _med_num_tokens_wi_rep)
#
#     def time_large_large_wo_rep(self):
#         simfunctions.overlap_coefficient(_large_num_tokens_wo_rep, _large_num_tokens_wo_rep)
#
#
#     def time_large_large_wi_rep(self):
#         simfunctions.overlap_coefficient(_large_num_tokens_wi_rep, _large_num_tokens_wi_rep)
#
#
#     def time_small_medium_wo_rep(self):
#         simfunctions.overlap_coefficient(_small_num_tokens_wo_rep, _med_num_tokens_wo_rep)
#
#     def time_small_medium_wi_rep(self):
#         simfunctions.overlap_coefficient(_small_num_tokens_wi_rep, _med_num_tokens_wi_rep)
#
#     def time_small_large_wo_rep(self):
#         simfunctions.overlap_coefficient(_small_num_tokens_wo_rep, _large_num_tokens_wo_rep)
#
#     def time_small_large_wi_rep(self):
#         simfunctions.overlap_coefficient(_small_num_tokens_wi_rep, _large_num_tokens_wi_rep)
#
#     def time_medium_large_wo_rep(self):
#         simfunctions.overlap_coefficient(_med_num_tokens_wo_rep, _large_num_tokens_wo_rep)
#
#     def time_medium_large_wi_rep(self):
#         simfunctions.overlap_coefficient(_med_num_tokens_wo_rep, _large_num_tokens_wo_rep)



