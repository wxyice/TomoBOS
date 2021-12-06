"""
ART 图像重建算法
可以把建立系数矩阵的过程放到两个变换的反函数中
2D
生成每个角度的系数值

用一个最大的范围去涵盖整个图像 sqrt(2*n^2)， 然后旋转中间的图像，然后每一行累积就可以了

借鉴清华论文的反函数训练方法，对整个函数做反函数训练
"""
import numpy as np

import os,shutil
from numpy.lib.shape_base import expand_dims
import scipy.io as scio
from matplotlib import pyplot as plt
from scipy import ndimage
from tqdm import tqdm


def Randon_Transformer(img,steps):
    '''
    单通道灰度雷登变换
    '''
    H,W=img.shape
    max_line=int(np.sqrt(H**2+W**2))+1
    field=np.zeros((max_line,max_line))
    f_H,f_W=field.shape

    field[(f_H//2-H//2):(f_H//2+H//2),(f_W//2-W//2):(f_W//2+W//2)]=img#photon_size
    
    photon=np.zeros((f_H,steps))

    for i in range(steps):
        field_rot=ndimage.rotate(field,i*(180/steps),reshape=False)
        Projection=np.sum(field_rot,axis=1)
        photon[:,i]=Projection
    
    return photon,field


def iRandon_by_DBP(photon):
    P,C=photon.shape
    photon=np.expand_dims(photon,axis=1)
    processing_P=np.repeat(photon,repeats=P,axis=1)/P
    processing_img=np.zeros((C,P,P))

    for i in range(C):
        processing_img[i,:,:]=ndimage.rotate(processing_P[:,:,i],-i*(180/steps),reshape=False)
    processing_img=np.average(processing_img,axis=0)
    return processing_img

def iRandon_by_Weight_DBP(photon,W):
    P,C=photon.shape
    photon=np.expand_dims(photon,axis=1)
    photon=np.transpose(photon,(2,0,1))
    #processing_P=np.repeat(photon,repeats=P,axis=1)/P

    processing_P=np.matmul(photon,W)
    processing_img=np.zeros((C,P,P))

    for i in range(C):
        processing_img[i,:,:]=ndimage.rotate(processing_P[i,:,:],-i*(180/steps),reshape=False)
        # fig,ax=plt.subplots(1,3)
        # ax[0].imshow(np.repeat(photon[i,:,:],repeats=P,axis=-1),cmap='gray')
        # ax[1].imshow(processing_P[i,:,:],cmap='gray')
        # ax[2].imshow(processing_img[i,:,:],cmap='gray')
        # plt.savefig('figtemp/{0}.jpg'.format(i))
        # plt.close()
    processing_img=np.average(processing_img,axis=0)
    return np.array(processing_img)

def RandonGAN(img,photon,steps):
    H,W=img.shape
    P,C=photon.shape

    max_line=int(np.sqrt(H**2+W**2))+1
    field=np.zeros((max_line,max_line))
    f_H,f_W=field.shape

    Weights=np.ones((C,1,P))/(P)
    photon_new=photon
    numiter=0

    mask=np.ones((H,W))
    ATA,_=Randon_Transformer(mask,steps=steps)
    #ATA=ATA[(f_H//2-H//2):(f_H//2+H//2),(f_W//2-W//2):(f_W//2+W//2)]
    ATA=iRandon_by_DBP(ATA)
    ATA=ATA[(f_H//2-H//2):(f_H//2+H//2),(f_W//2-W//2):(f_W//2+W//2)]
    # plt.imshow(ATA,cmap='gray')
    # plt.show()
    while True:
        numiter+=1
        #processing_img=iRandon_by_Weight_DBP(photon,Weights)

        processing_img=iRandon_by_DBP(photon_new)
        processing_img=processing_img[(f_H//2-H//2):(f_H//2+H//2),(f_W//2-W//2):(f_W//2+W//2)]
        photon_new,_=Randon_Transformer(processing_img,steps=steps)
        err=photon-photon_new


        print(np.sum(err**2))
        #err_img=src_img-processing_img

        err_img=iRandon_by_DBP(err)
        err_img=err_img[(f_H//2-H//2):(f_H//2+H//2),(f_W//2-W//2):(f_W//2+W//2)]

        fig,ax=plt.subplots(2,3)
        ax[0][0].imshow(src_img,cmap='gray')
        ax[0][1].imshow(processing_img,cmap='gray')
        ax[0][2].imshow(err_img,cmap='gray')
        ax[1][0].imshow(photon,cmap='gray')
        ax[1][1].imshow(photon_new,cmap='gray')
        ax[1][2].imshow(err,cmap='gray')
        plt.savefig('fig222/res_{}_err_{}.jpg'.format(numiter,np.sum(err**2)))
        plt.close()

        if np.sum(err**2)<1e-5:
            break

        processing_img=processing_img+0.1*err_img/ATA
        #processing_img=processing_img[(f_H//2-H//2):(f_H//2+H//2),(f_W//2-W//2):(f_W//2+W//2)]
        photon_new,_=Randon_Transformer(processing_img,steps=steps)

        #photon=photon_new
        # fig,ax=plt.subplots(1,2)
        # ax[0].imshow(np.repeat(Weights[0,:,:],repeats=P,axis=0),cmap='gray')
        # ax[1].imshow(np.repeat(np.expand_dims(err[:,0],axis=-1),repeats=P,axis=1),cmap='gray')
        # plt.show()

        # err=np.expand_dims(err,axis=1)
        # err=np.transpose(err,(2,1,0))
        # fig,ax=plt.subplots(1,2)
        # ax[0].imshow(np.repeat(Weights[0,:,:],repeats=P,axis=0),cmap='gray')
        # ax[1].imshow(np.repeat(np.expand_dims(err[:,0],axis=-1),repeats=P,axis=1),cmap='gray')
        # plt.show()
        # Weights-=err

        # Min=np.min(Weights,axis=-1)
        # Min=np.expand_dims(Min,axis=-1)
        # Min=np.repeat(Min,repeats=P,axis=-1)

        # Max=np.max(Weights,axis=-1)
        # Max=np.expand_dims(Max,axis=-1)
        # Max=np.repeat(Max,repeats=P,axis=-1)

        # S=np.sum(Weights,axis=-1)
        # S=np.expand_dims(S,axis=-1)
        # S=np.repeat(S,repeats=P,axis=-1)

        # Weights=(Weights-Min)/(Max-Min)/S
        # fig,ax=plt.subplots(1,2)
        # ax[0].imshow(np.repeat(Weights[0,:,:],repeats=P,axis=0),cmap='gray')
        # ax[1].imshow(np.repeat(np.expand_dims(photon[:,0],axis=-1),repeats=P,axis=1),cmap='gray')
        # plt.show()







if __name__ == '__main__':

    root='test/ART_test'
    import datetime
    now_time = str(datetime.datetime.now().strftime('%Y%m%d %H%M%S'))
    
    try:
        os.mkdir(root)
        os.mkdir(os.path.join(root,now_time))
    except:
        pass

    path=r'processing_data\XCAT512.mat'
    src_img = scio.loadmat(path)['XCAT512']
    steps=10

    photon,field=Randon_Transformer(src_img,steps)

    RandonGAN(src_img,photon,steps)
    #photon=iRandon_by_DBP(photon)


    # fig,ax=plt.subplots(1,2)
    # ax[0].imshow(photon,cmap='gray')
    # ax[1].imshow(field,cmap='gray')
    # plt.show()
