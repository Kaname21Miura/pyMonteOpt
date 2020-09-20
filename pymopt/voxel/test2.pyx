
"""
Created on Thu Sep 17 20:12:59 2020

@author: kaname
"""

import numpy as np
cimport numpy as np
np.import_array()

from libc.stdlib cimport rand, RAND_MAX
    

cpdef float random_uniform():
    cdef float random = float(rand())
    cdef float randmax = float(RAND_MAX)
    return random/randmax
