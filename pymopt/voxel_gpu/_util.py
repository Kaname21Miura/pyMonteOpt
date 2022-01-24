#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 17:02:21 2021
@author: Kaname Miura
"""

import pycuda.driver as cuda

def _get_device_config():
    aa =0
    for i in range(cuda.Device.count()):
        dev=cuda.Device(i)
        aa = dev.get_attributes()

    for i in aa:
        print(i,": ",aa[i])
