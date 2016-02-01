import functools
import numpy as np

"""
This module defines a list of decorator functions to check input strings/list. The reason this is separated
from the similarity functions is the implementation of checking functions can change later, depending on
our decision to handle missing values.
"""


def sim_check_for_same_len(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        if args[0] is None:
            raise TypeError("string1 is None")
        if args[1] is None:
            raise TypeError("string2 is None")
        if len(args[0]) != len(args[1]):
            raise ValueError("Undefined for sequences of unequal length")
        return func(*args, **kwargs)
    return decorator


def sim_check_for_exact_match(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        if args[0] == args[1]:
            return 1.0
        return func(*args, **kwargs)
    return decorator


def sim_check_for_empty(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        if len(args[0]) == 0 or len(args[1]) == 0:
            return 0
        return func(*args, **kwargs)
    return decorator


def sim_check_for_none(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        if args[0] is None:
            raise TypeError("string1 is None")
        if args[1] is None:
            raise TypeError("string2 is None")
        return func(*args, **kwargs)
    return decorator


def tok_check_for_none(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        empty_list = []
        if args[0] is None:
            return empty_list
        return func(*args, **kwargs)
    return decorator







# # check for NaNs
# def check_strings_for_nulls(func):
#     @functools.wraps(func)
#     def decorator(*args, **kwargs):
#         if np.isnan(args[0]) is True:
#             return np.NaN
#         if np.isnan(args[1]) is None:
#             return np.NaN
#         return func(*args, **kwargs)
#     return decorator
#
# # check for nulls in tokens
# def check_tokens_for_nulls(func):
#     @functools.wraps(func)
#     def decorator(*args, **kwargs):
#         tmp_args0 = args[0]
#         if not isinstance(tmp_args0, list):
#             tmp_args0 = [tmp_args0]
#         if any(np.isnan(tmp_args0)) is True:
#             return np.NaN
#         tmp_args1 = args[1]
#         if not isinstance(tmp_args1, list):
#             tmp_args1 = [tmp_args1]
#         if any(np.isnan(tmp_args1)) is True:
#             return np.NaN
#         return func(*args, **kwargs)
#     return decorator


