#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:08:37 2020

@author: kaname
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

sourcefiles = ['_voxel_montecarlo.pyx']
setup(
    cmdclass = {'build_ext':build_ext},
    ext_modules = [Extension('_voxel_montecarlo', sourcefiles)],
    include_dirs = [np.get_include()]
)