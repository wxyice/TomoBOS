"""
ART 图像重建算法
可以把建立系数矩阵的过程放到两个变换的反函数中
2D
生成每个角度的系数值

用一个最大的范围去涵盖整个图像 sqrt(2*n^2)， 然后旋转中间的图像，然后每一行累积就可以了

借鉴清华论文的反函数训练方法，对整个函数做反函数训练
"""
import os
import shutil

import numpy as np
import scipy.io as scio
from cv2 import cv2
from matplotlib import pyplot as plt
from numpy.lib.shape_base import expand_dims
from scipy import ndimage
from tqdm import tqdm


def Randon_Transformer(img,steps):
    '''
    单通道灰度雷登变换
    '''
    H,W=img.shape
    photon=np.zeros((W,steps))
    for i in range(steps):
        field_rot=ndimage.rotate(img,i*(180/steps),reshape=False)
        Projection=np.sum(field_rot,axis=1)
        photon[:,i]=Projection
    return photon

def iRadon_by_DBP(photon):
    P,C=photon.shape
    photon=np.expand_dims(photon,axis=1)
    processing_P=np.repeat(photon,repeats=P,axis=1)/P
    processing_img=np.zeros((C,P,P))

    for i in range(C):
        processing_img[i,:,:]=ndimage.rotate(processing_P[:,:,i],-i*(180/steps),reshape=False)
    processing_img=np.average(processing_img,axis=0)
    return processing_img

def RadonIterSolver(img,photon,steps,save_path):

    show=False
    H,W=img.shape

    numiter=0

    mask=np.ones((H,W))
    ATA=Randon_Transformer(mask,steps=steps)
    ATA=iRadon_by_DBP(ATA)

    processing_img=iRadon_by_DBP(photon)

    while True:
        numiter+=1

        photon_new=Randon_Transformer(processing_img,steps=steps)
        photon_err=photon-photon_new
        err_img=iRadon_by_DBP(photon_err)

        print(np.sum(photon_err**2))
        if np.sum(photon_err**2)<1e-5:
            break
        processing_img=processing_img+1.2*err_img/ATA

        if show:
            cv2.imshow('rrr',((processing_img-np.min(processing_img))/(np.max(processing_img)-np.min(processing_img))*255).astype(np.uint8))
            cv2.waitKey(1)
        

        save_fig(src_img,processing_img,err_img,photon,photon_new,photon_err,numiter,save_path)

def save_fig(src_img,processing_img,err_img,photon,photon_new,photon_err,numiter,save_path):
    fig,ax=plt.subplots(2,3)
    ax[0][0].imshow(src_img,cmap='gray')
    ax[0][1].imshow(processing_img,cmap='gray')
    ax[0][2].imshow(err_img,cmap='gray')
    ax[1][0].imshow(photon,cmap='gray')
    ax[1][1].imshow(photon_new,cmap='gray')
    ax[1][2].imshow(photon_err,cmap='gray')
    plt.savefig(os.path.join(save_path,'res_{}_err_{}.jpg'.format(numiter,np.sum(photon_err**2))))
    plt.close()

if __name__ == '__main__':

    root='test/ART_test'
    import datetime
    now_time = str(datetime.datetime.now().strftime('%Y%m%d %H%M%S'))
    
    try:
        os.mkdir(root)
        os.mkdir(os.path.join(root,now_time))
    except:
        pass

    savepath=os.path.join(root,now_time)

    path=r'processing_data\XCAT512.mat'
    src_img = scio.loadmat(path)['XCAT512']
    steps=10

    photon=Randon_Transformer(src_img,steps)

    RadonIterSolver(src_img,photon,steps,savepath)
