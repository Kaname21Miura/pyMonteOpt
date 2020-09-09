#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 17:14:26 2020

@author: kaname
"""
import numpy as np
import time
#import itertools
import IntarnalFluence


class MonteCalro(object):
    def __init__(self,*initial_data, **kwargs):
        self.keys = ['nPh','g','ma','ms','n','n_air','thckness','fluence','f_bit','vectorTh']
        self.nPh = 1000
        self.f_bit = 'float32'
        self.vectorTh = 0.99999
        self.fluence = False
        
        self.borderposit = np.array([0])
        self.thckness = np.array([0])
        self.n = np.array([1])
        self.n_air = 1
        self.ref_index = np.array([1,1])
        self.ms = np.array([1])
        self.ma = np.array([1])
        self.g = np.array([1])
        
        self.setParams(*initial_data, **kwargs)
        
        self.v_result = np.empty((3,1)).astype(self.f_bit)
        self.p_result = np.empty((3,1)).astype(self.f_bit)
        self.w_result = np.empty(1).astype(self.f_bit)
        
        self.p = np.empty((3,1)).astype(self.f_bit)
        self.v = np.empty((3,1)).astype(self.f_bit)
        self.w = np.empty(1).astype(self.f_bit)
        self.add = np.empty((3,1)).astype(self.f_bit)
        
    def checkNumpyArray(self,val,key):
        if (not type(val) is np.ndarray)or(not type(val) is list):
            e_mess = key + ' must be list or ndarray'
            raise ValueError(e_mess)
            
    def checkPrams(self):
        check_values = np.array([self.g,self.ma,self.ms,self.n,self.thckness])
        check_keys = ['g','ma','ms','n','thckness']
        for val,key in zip(check_values,check_keys):
            self.checkNumpyArray(val,key)
            setattr(self,key,np.array(val).astype(self.f_bit))
        
    def setParams(self,*initial_data, **kwargs):
        def specialKey(key,item):
            if key == 'fluence':
                self.setFluenceClass(item)  
                
        for dictionary in initial_data:
            for key in dictionary:
                if not key in self.keys:
                    raise KeyError(key)
                setattr(self,key, dictionary[key])
                specialKey(key, dictionary[key])
        for key in kwargs:
            if not key in self.keys:
                raise KeyError(key)
            setattr(self, key, kwargs[key])
            specialKey(key, kwargs[key])
            
        self.setBorderPosit()
        self.checkPrams()
        self.setRefIndex()
        
    def setFluenceClass(self,flue):
        if isinstance(flue,IntarnalFluence):
            setattr(self, 'fluence', flue)
        elif flue == False:
            setattr(self, 'fluence', flue)
        else:
            raise KeyError('fluence should be input fluence class')

    def setRefIndex(self):
        border = np.append(self.n_air,self.n)
        border = np.append(border,self.n_air).astype(self.f_bit)
        setattr(self,'borderposit',border)
        
    def setBorderPosit(self):
        thick = [0]+self.thickness
        b = 0; b_list = []
        for i in  thick:
            b += i
            b_list.append(b)
        setattr(self,'ref_index',np.array(b_list).astype(self.f_bit))
    
    def monteCycle(self,start_):
        count = 0
        counter = 2
        w_size = 1
        # Let's MonteCalro!
        while w_size == 0:
            w_size = self.stepMovement()
            count+=1
            if count%counter==0:
                counter*=2
                print("Progress: %s [％]"%round((1-w_size/self.nPh)*100,3))
                self.calTime(time.time(), start_)
                print()
        return count
    
    def startMonteCalro(self,nPh):
        print("")
        print("###### Start ######")
        print("")
        
        start_ = time.time()
        #初期値の設定
        self.generateInisalCoodinate(nPh)
        count = self.monteCycle(start_)

        #結果の表示
        print("")
        print("###### Finish ######")
        print("Maximum step number: %s"%count)
        self.calTime(time.time(), start_)
        
    def initialWeight(self,w):
        Rsp = 0
        n1 = self.ref_index[0]
        n2 = self.ref_index[1]
        if n1 != n2:
            Rsp = ((n1-n2)/(n1+n2))**2
        return w-Rsp
    
    def vectorUpdate(self,v,G,f="float32"):
        index = np.where(G==0.0)[0]
        cosTh = np.empty_like(G)
        if list(index) != []:
            rand_num = np.random.rand(G.size).astype(f)
            cosTh[index] = 2*rand_num[index]-1
            index = np.where(G!=0)[0]
            if list(index) != []:
                cosTh[index] = (
                    (1+G[index]**2-((1-G[index]**2)/(1-G[index]+2*G[index]*rand_num[index]))**2
                     )/(2*G[index])
                        )
        else:
            cosTh = (1+G**2-((1-G**2)/(1-G+2*G*np.random.rand(G.size).astype(f)))**2)/(2*G)
        sinTh = np.sqrt(1-cosTh**2)
        
        #cos(fai)とsin(fai)と求める
        Fi = 2*np.pi*np.random.rand(G.size).astype(f)
        cosFi = np.cos(Fi)
        sinFi = np.sin(Fi)
        
        #Zが１かそれ以外で分離
        th = self.vectorTh
        v1_index = np.where(np.abs(v[2])<=th)[0]
        v2_index = np.where(np.abs(v[2])>th)[0]
        
        #Z方向ベクトルが0.99999以下
        v1 = v[:,v1_index]
        cosTh1 = cosTh[v1_index]; sinTh1 = sinTh[v1_index]
        cosFi1 = cosFi[v1_index]; sinFi1 = sinFi[v1_index]
        B = np.sqrt(1-v1[2]**2)
        A = np.array([
            sinTh1*(v1[0]*v1[2]*cosFi1-v1[1]*sinFi1)/B,
            sinTh1*(v1[1]*v1[2]*cosFi1+v1[0]*sinFi1)/B,
            -sinTh1*cosFi1*B,
        ])
        v[:,v1_index] = A+v1*cosTh1
            
        #Z方向ベクトルが0.99999以上
        v2 = v[:,v2_index]
        cosTh2 = cosTh[v2_index]; sinTh2 = sinTh[v2_index]
        cosFi2 = cosFi[v2_index]; sinFi2 = sinFi[v2_index]
        v[:,v2_index] = np.array([
            sinTh2*cosFi2,
            sinTh2*sinFi2,
            np.sign(v2[2])*cosTh2,
        ],dtype=f)
            
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
            
    def russianRoulette(self,w,f="float32"):
        ## 確率的に光子を生き返らせます。
        m = 10
        ra = np.random.rand(w.size).astype(f)
        index = np.where(ra>(1/m))[0].tolist()
        w[index] = 0
        index = np.where(ra<=(1/m))[0].tolist()
        w[index] = w[index]*m
        return w

    #光子の移動距離, uniformly distributed over the interval (0,1)
    def stepLength(self,size,f="float32"):
        return -np.log(np.random.rand(size)).astype(f)
    
    #任意の位置(indexの行)が１でそれ以外は0の行列を作る
    def create01Array(self,index,m=3):
        n = index.size
        array_0_1 = np.zeros(m*n,dtype = bool)
        array_0_1[index+m*np.arange(n)] = 1
        return array_0_1.reshape(n,m).T 
    
    
    ##### モデルによってオーバーライドして使う関数 #####
    #光の動きを制御する
    def stepMovement(self):
        return self.w.size
    
    def generateInisalCoodinate(self,nPh):
        self.p = np.zeros((3,nPh),dtype = self.f_bit)
        self.v = np.zeros((3,nPh),dtype = self.f_bit)
        self.v[2] = 1
        self.w = self.initialWeight(np.full(nPh,1).astype(self.f_bit))
    
    def saveResult(self,result):
        if result[0][0].tolist() != []:
            self.p_result = np.concatenate([self.p_result, result[0]],axis = 1)
            self.v_result = np.concatenate([self.v_result, result[1]],axis = 1)
            self.w_result = np.concatenate([self.w_result, result[2]])
            
    def getResult(self):
        self.v_result = self.v_result[:,1:]
        self.p_result = self.p_result[:,1:]
        self.w_result = self.w_result[1:]
        result = {
            'p':self.p_result,
            'v':self.v_result,
            'w':self.w_result,
        }
        return result

