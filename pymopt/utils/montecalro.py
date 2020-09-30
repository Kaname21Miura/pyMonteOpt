#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 17:14:26 2020

@author: kaname
"""
import numpy as np
import time
import warnings
from collections import defaultdict
import inspect
from ..utils.utilities import calTime

class MonteCalro:
    def __init__(self):
        self.nPh = 1000
        self.f_bit = 'float32'
        self.vectorTh = 0.99999

        self.v_result = np.empty((3,1)).astype(self.f_bit)
        self.p_result = np.empty((3,1)).astype(self.f_bit)
        self.add_result = np.empty((3,1)).astype('int16')
        self.w_result = np.empty(1).astype(self.f_bit)

        self.p = np.empty((3,1)).astype(self.f_bit)
        self.v = np.empty((3,1)).astype(self.f_bit)
        self.w = np.empty(1).astype(self.f_bit)

        self.Rdw = 0
        self.Ttw = 0

    def start(self):
        print("")
        print("###### Start ######")
        print("")

        start_ = time.time()

        count = self.monteCycle(start_)
        self.endProcess()

        #結果の表示
        print("")
        print("###### Finish ######")
        print("Maximum step number: %s"%count)
        self.getRdTtRate()
        calTime(time.time(), start_)
        return self

    def monteCycle(self,start_):
        count = 0
        counter = 2
        w_size = 1
        # Let's MonteCalro!
        while w_size != 0:
            w_size = self.stepMovement()
            count+=1
            if count%counter==0:
                counter*=2
                print("Progress: %s [％]"%round((1-w_size/self.nPh)*100,3))
                calTime(time.time(), start_)
                print()
        return count


    def endProcess(self):
        pass

    def vectorUpdate(self,v,G):
        index = np.where(G==0.0)[0]
        cosTh = np.empty_like(G)
        if list(index) != []:
            rand_ = np.random.rand(G.size).astype(self.f_bit)
            cosTh[index] = 2*rand_[index]-1
            index = np.where(G!=0)[0]
            if list(index) != []:
                cosTh[index] = ((1+G[index]**2\
                                 -((1-G[index]**2)/(1-G[index]+2*G[index]*rand_[index]))**2)/(2*G[index]))
        else:
            cosTh = (1+G**2-((1-G**2)/(1-G+2*G*np.random.rand(G.size).astype(self.f_bit)))**2)/(2*G)
        sinTh = np.sqrt(1-cosTh**2)

        #cos(fai)とsin(fai)と求める
        Fi = 2*np.pi*np.random.rand(G.size).astype(self.f_bit)
        cosFi = np.cos(Fi)
        sinFi = np.sin(Fi)

        #Zが１かそれ以外で分離
        th = self.vectorTh
        v1_index = np.where(np.abs(v[2])<=th)[0]
        v2_index = np.where(np.abs(v[2])>th)[0]

        #Z方向ベクトルが0.99999以下
        cosTh1 = cosTh[v1_index]; sinTh1 = sinTh[v1_index]
        cosFi1 = cosFi[v1_index]; sinFi1 = sinFi[v1_index]
        v1 = v[:,v1_index]
        B = np.sqrt(1-v1[2]**2)
        A = np.array([
            sinTh1*(v1[0]*v1[2]*cosFi1-v1[1]*sinFi1)/B,
            sinTh1*(v1[1]*v1[2]*cosFi1+v1[0]*sinFi1)/B,
            -sinTh1*cosFi1*B,
        ])
        v[:,v1_index] = A+v1*cosTh1

        #Z方向ベクトルが0.99999以上
        cosTh2 = cosTh[v2_index]; sinTh2 = sinTh[v2_index]
        cosFi2 = cosFi[v2_index]; sinFi2 = sinFi[v2_index]
        v[:,v2_index] = np.array([
            sinTh2*cosFi2,
            sinTh2*sinFi2,
            np.sign(v[2,v2_index])*cosTh2,
        ],dtype=self.f_bit)

        v = v/np.linalg.norm(v,axis=0)
        return v

    def positionUpdate(self,p,v,L):
        return p+v*L

    #光子の１ステップにおけるエネルギーの損失を計算
    def wUpdate(self,w,ma,mt,rato,p):
        dw = w*rato*ma/mt
        if self.fluence != False:
            self.fluence.saveFluesnce(p,dw)
        return w-dw

    def russianRoulette(self,w):
        ## 確率的に光子を生き返らせます。
        m = 10
        ra = np.random.rand(w.size).astype(self.f_bit)
        index = np.where(ra>(1/m))[0].tolist()
        w[index] = 0
        index = np.where(ra<=(1/m))[0].tolist()
        w[index] = w[index]*m
        return w

    #光子の移動距離, uniformly distributed over the interval (0,1)
    def stepLength(self,size):
        return -np.log(np.random.rand(size)).astype(self.f_bit)

    #任意の位置(indexの行)が１でそれ以外は0の行列を作る
    def create01Array(self,index,m=3):
        n = index.size
        array_0_1 = np.zeros(m*n,dtype = bool)
        array_0_1[index+m*np.arange(n)] = 1
        return array_0_1.reshape(n,m).T

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("monte method should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def set_params(self,**params):
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `monte.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def get_params(self, deep=True):

        out = dict()
        for key in self._get_param_names():
            try:
                value = getattr(self, key)
            except AttributeError:
                warnings.warn('From version 0.24, get_params will raise an '
                              'AttributeError if a parameter cannot be '
                              'retrieved as an instance attribute. Previously '
                              'it would return None.',
                              FutureWarning)
                value = None
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out


    def getRdTtRate(self):
        self.Tt_index = np.where(self.v_result[2]>0)[0]
        self.Rd_index = np.where(self.v_result[2]<0)[0]
        self.Rdw = self.w_result[self.Rd_index].sum()/self.nPh
        self.Ttw = self.w_result[self.Tt_index].sum()/self.nPh
        print('######')
        print('Mean Rd %0.6f'% self.Rdw)
        print('Mean Tt %0.6f'% self.Ttw)
        print()

    def getRdTtIndex(self):
        return {
            'Rd':self.Rd_index,
            'Tt':self.Tt_index,
        }

    def getRdTtValues(self):
        return {
            'Rd':self.Rdw,
            'Tt':self.Ttw,
        }
