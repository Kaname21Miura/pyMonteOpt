#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 15:45:55 2020

@author: kaname
"""

from functools import wraps
from inspect import signature, Parameter #,isclass
import warnings

def _deprecate_positional_args(f):
    """Decorator for methods that issues warnings for positional arguments.
    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning when passed as a positional argument.
    Parameters
    ----------
    f : callable
        Function to check arguments on.
    """
    sig = signature(f)
    kwonly_args = []
    all_args = []

    for name, param in sig.parameters.items():
        if param.kind == Parameter.POSITIONAL_OR_KEYWORD:
            all_args.append(name)
        elif param.kind == Parameter.KEYWORD_ONLY:
            kwonly_args.append(name)

    @wraps(f)
    def inner_f(*args, **kwargs):
        extra_args = len(args) - len(all_args)
        if extra_args <= 0:
            return f(*args, **kwargs)

        # extra_args > 0
        args_msg = ['{}={}'.format(name, arg)
                    for name, arg in zip(kwonly_args[:extra_args],
                                         args[-extra_args:])]
        warnings.warn("Pass {} as keyword args. From version 0.25 "
                      "passing these as positional arguments will "
                      "result in an error".format(", ".join(args_msg)),
                      FutureWarning)
        kwargs.update(zip(sig.parameters, args))
        return f(**kwargs)
    return inner_f