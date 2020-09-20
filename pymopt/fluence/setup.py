#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:08:37 2020

@author: kaname
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize("_fluence.pyx"),
    include_dirs = [np.get_include()]
)