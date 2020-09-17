#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 21:56:21 2020

@author: kaname
"""

import numpy as np

def compute(loop_a, loop_b):
    result = 0

    for a in range(loop_a):
        for b in range(loop_b):
            result += 1
    return result