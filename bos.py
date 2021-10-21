#----------------------------------------------
# Date: 2021.10.14
# Author: Wxyice
# Funtion：
#   
#----------------------------------------------

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import scipy.io as scio
import SimpleITK as sitk
from cv2 import cv2
from icecream import ic
from numpy.lib.financial import npv
from PIL import Image

from DBP import FBPIRandonTransform
from show3D import plot_3d


def normalize(img):
    ratio=(np.max(img)-np.min(img))
    bias=np.min(img)
    img=(img-bias)/ratio
    img=img*255
    img=img.astype(np.uint8)
    return img,ratio,bias

class BOS():
    def __init__(self,ZD,ZA,f,n0) -> None:
        self.ZD=ZD
        self.ZA=ZA
        self.f=f
        self.n0=n0
        pass

    def Filterbackporcessing(self):
        pass
    
    

if __name__ == '__main__':
    ic.disable()

    bos01=scio.loadmat('01.mat')

    print(bos01.keys())

    delta_magnitude=bos01['velocity_magnitude']
    print(delta_magnitude[0][0].shape)

    photon=delta_magnitude[0][0]# shape(101,104)x,y
    photon,ratio,bias=normalize(photon)


    #photon=cv2.resize(photon,(photon.shape[1]*5,photon.shape[0]*5))
    # cv2.imshow('p',photon)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #print(photon)

    path_save_img='01'
    img3D=np.zeros((photon.shape[0],photon.shape[1],photon.shape[1]))
    imgframe=[]
    if os.path.exists(path_save_img)!=True:
        os.mkdir(path_save_img)
    
    for i in range(photon.shape[0]):
        line=photon[i,:]  # 取出一行
        line=np.expand_dims(line,axis=1)
        line=np.repeat(line,360,axis=1)
        #print(line.shape)
        #cv2.imshow('a',line)

        img_back=FBPIRandonTransform(line,len(line[0]),'SL',show=False)
        img_back,ratio,bias=normalize(img_back)
        cv2.imwrite(os.path.join(path_save_img,'{0:03d}.png'.format(i)),img_back)
        imgframe.append(img_back)
        img3D[i,:,:]=img_back
        #print(img_back.shape)
        img_back=cv2.resize(img_back,(img_back.shape[1]*5,img_back.shape[0]*5))
        cv2.imshow('p',img_back)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

    i=0
    while i<photon.shape[-1]:
        
        img=img3D[:,:,i].astype(np.uint8)
        print(img.shape,img)
        img=cv2.resize(img,(img.shape[1]*5,img.shape[0]*5))
        cv2.imshow('rrr',img)
        cv2.waitKey(0)
        i+=1        

    cv2.destroyAllWindows()

    print(img3D)

    delta_3D=img3D*ratio+bias


    # sitk_img = sitk.GetImageFromArray(img3D, isVector=False)
    # sitk_img.SetSpacing(ConstPixelSpacing)
    # sitk_img.SetOrigin(Origin)
    # print(sitk_img)
    # sitk.WriteImage(sitk_img, os.path.join(SaveRawDicom, "sample" + ".mhd"))
