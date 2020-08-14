# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:08:54 2019

@author: Kaname Miura
"""

import numpy as np
import pydicom as dicom
import os
import matplotlib.pyplot as plt

workspace = os.getcwd()
os.chdir(workspace)

def readDicom(pathdcom):
    os.chdir(workspace)
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(pathdcom):
        for filename in fileList:
            if ".dcm" in filename.lower(): # 拡張子が.dcmか.magかで書き換える
                lstFilesDCM.append(os.path.join(dirName,filename))
    
    RefDs = dicom.read_file(lstFilesDCM[0],force=True) # DICOMの先頭ファイルはヘッダとなる
    RefDs.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
    
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.PixelSpacing[1]))
    x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
    y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
    z = np.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])
    
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
    
    # すべてのDICOMファイルに対して読み込む
    for filenameDCM in lstFilesDCM:
        ds = dicom.read_file(filenameDCM,force=True)
        ds.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
        ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array
    
    ArrayDicom = changeResolution(ArrayDicom,ConstPixelDims)
    print("ConstPixelDims: %s"%str(ConstPixelDims))
    print("ConstPixelSpacing: %s"%str(ConstPixelSpacing))
    print("Data infomation")
    print(RefDs)
    os.chdir(workspace)
    return ArrayDicom,[x,y,z],ConstPixelDims,ConstPixelSpacing
    
def changeResolution(x,ConstPixelDims):
    #解像度を16bitから8bitに変更します。
    a = x.shape
    x = x.reshape(-1,ConstPixelDims[2])
    x = np.array([np.round(i/(2**8)).astype("int8") for i in x])
    return np.array(x).reshape(a[0],a[1],a[2])

def reConstArray(ArrayDicom,threshold=9500):
    threshold = round(threshold/(2**8))
    return np.where(ArrayDicom < threshold, 0, ArrayDicom)

def displayGraph(ArrayDicom,resolution):
    plt.figure(figsize=(10,6),dpi=200)
    plt.axes().set_aspect('equal', 'datalim')
    plt.set_cmap(plt.gray())
    plt.pcolormesh(resolution[1], resolution[0], ArrayDicom[:, :,0])
    plt.xlabel("mm")
    plt.ylabel("mm")
    plt.show()
    
# 表示

#pathdcom = "DICOMfile9" 
#ArrayDicom,resolution,ConstPixelDims = readDicom(pathdcom)
#ArrayDicom = reConstArray(ArrayDicom)

             
