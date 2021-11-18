"""
# Date: 2021.10.14
# Author: Wxyice
# Funtion：
#   
"""

import os
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import scipy.io as scio
import SimpleITK as sitk
from cv2 import GaussianBlur, cv2
from icecream import ic
from numpy.core.fromnumeric import shape
from numpy.lib.financial import npv
from openpiv import filters, pyprocess, scaling, tools, validation
from PIL import Image
from tqdm import tqdm

from DBP import FBPIRandonTransform, LBPIRandonTransform
from Poissonsolve2D import buildMatrix
from show3D import plot_3d
from solve import CG, SD, Gauss_Seidel


def normalize(img):
    ratio=(np.max(img)-np.min(img))
    bias=np.min(img)
    img=(img-bias)/ratio
    img=img*255
    img=img.astype(np.uint8)
    return img,ratio,bias

class BOS():
    """
        BOS class
        to process BOS image
        Input:
        :param: ZD
        :param: ZA
        :param: f
        :param: n0
    """
    def __init__(self,ZD,ZA,f,n0) -> None:
        # 常数
        G=2.2244*1e-4
        self.ZD=ZD
        self.ZA=ZA
        self.f=f
        self.n0=n0
        self.C=(self.n0/G)*((self.ZD+self.ZA-f)/(self.ZD*f))
        self.C=(self.n0)*((self.ZD+self.ZA-f)/(self.ZD*f))
    
    def cross_correlation(self,path_a,path_b,winsize=32,searchsize=38,overlap=16,dt=1,show=False):
        """
        计算图片互相关算法，计算a，b两张图，a为前，b为后
        :param: winsize     # pixels, interrogation window size in frame A
        :param: searchsize  # pixels, search area size in frame B
        :param: overlap     # pixels, 50% overlap
        :param: dt          # sec, time interval between the two frames
        """
        a=tools.imread(path_a)
        b=tools.imread(path_b)
        
        u0, v0, sig2noise = pyprocess.extended_search_area_piv(
            a.astype(np.int32),
            b.astype(np.int32),
            window_size=winsize,
            overlap=overlap,
            dt=dt,
            search_area_size=searchsize,
            sig2noise_method='peak2peak',
        )
        x, y = pyprocess.get_coordinates(
            image_size=a.shape,
            search_area_size=searchsize,
            overlap=overlap,
        )
        u1, v1, mask = validation.sig2noise_val(
            u0, v0,
            sig2noise,
            threshold = 1.05,
        )
        u2, v2 = filters.replace_outliers(
            u1, v1,
            method='localmean',
            max_iter=3,
            kernel_size=3,
        )
        # convert x,y to mm
        # convert u,v to mm/sec
        x, y, u3, v3 = scaling.uniform(
            x, y, u2, v2,
            scaling_factor = 1056/1000,  # 96.52 pixels/millimeter  # 标定因子
        )
        # 0,0 shall be bottom left, positive rotation rate is counterclockwise
        x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)
        if show:
            fig, ax = plt.subplots(figsize=(8,8))
            tools.save(x, y, u3, v3, mask, 'exp1_001.txt' )
            tools.display_vector_field(
                'exp1_001.txt',
                ax=ax, scaling_factor=1056/100,
                scale=8, # scale defines here the arrow length
                width=0.0035, # width is the thickness of the arrow
                on_img=True, # overlay on the image
                image_name=path_a,
            )
        return x,y,u3,v3

    def Filterbackporcessing(self,photon,show=False,save_path=None):
        """
            基于轴对称假设的反投影方法
            photon 幻影图像 二维 shape=2
        """
        H,W=photon.shape
        img3D=np.zeros((H,W,W))

        for i in tqdm(range(H)):
            line=photon[i,:]  # 取出一行
            line=np.expand_dims(line,axis=1)
            line=np.repeat(line,180,axis=1)

            img_back=LBPIRandonTransform(line,len(line[0]))#,'SL',show=False)
            img3D[i,:,:]=img_back
            # print('processing {0} in {1}'.format(i,photon.shape[0]))
            if show:
                plt.ion()
                plt.imshow(img_back,cmap='gray')
                plt.pause(1)
                plt.close()
        if save_path!=None:
            np.save(save_path, img3D)
        return img3D
    
    def load_from_mat(self,path):
        bos01=scio.loadmat(path)
        self.delta_magnitude=bos01['velocity_magnitude']
        self.delta_u,self.delta_v=bos01['u_original'], bos01['v_original']
        self.x,self.y=bos01['x'],bos01['y']
    
    def slice_show(self,img3D):
        i=0
        while i<img3D.shape[-1]:
            
            img=img3D[:,:,i]
            print(img.shape,img)
            plt.ion()
            plt.imshow(img,cmap='gray')
            plt.pause(1)
            plt.close()
            i+=1        

    def solve_rho(self,x,y,Dx,Dy):
        """
            求解密度场
        """
        # x,y,Dx,Dy，xy是坐标位置矩阵，Dx，Dy应该来自于反演后的结果，所得到的B结果还需要一个系数比例C
        # 去除nan信息
        Dmask=np.isnan(Dx)
        Dx[Dmask]=0
        Dmask=np.isnan(Dy)
        Dy[Dmask]=0

        A,B=buildMatrix(x,y,Dx,Dy,self.C)
        print(np.max(B),np.min(B))
        H,W=x.shape
        #B=self.C*B

        x=CG(A,B)
        b=A @ x
        print(A@x)
        #x=SD(A,B,maxiter=3000)
        # x=Gauss_Seidel(A,B)

        x=x.reshape((H,W))
        return x        

if __name__ == '__main__':

    # img3D 的格式[z,x,y]
    ic.disable()
    ZD=0.5
    ZA=0.5
    f=0.05
    n0=1

    mat_path=r'processingdata\01_PIVprocess.mat'

    Bos_pipeline=BOS(ZD,ZA,f,n0)

    # PIVLab + FBP 
    Bos_pipeline.load_from_mat(mat_path)
    photon=Bos_pipeline.delta_magnitude[0][0]
    x=Bos_pipeline.x[0][0]
    y=Bos_pipeline.y[0][0]
    u=Bos_pipeline.delta_u[0][0]
    v=Bos_pipeline.delta_v[0][0]
    #img3D=Bos_pipeline.Filterbackporcessing(photon)

    # openPIV
    # path1=r'raw2jpg\back.jpg'
    # path2=r'raw2jpg\test1.jpg'

    # winsize=64   #20
    # searchsize=64#20
    # overlap=32   #10

    # x,y,u,v=Bos_pipeline.cross_correlation(path_a=path1,path_b=path2,winsize=winsize,searchsize=searchsize,overlap=overlap,show=False)

    # mask=np.isnan(u)
    # u[mask]=0
    # mask=np.isnan(v)
    # v[mask]=0
    # print(np.max(u),np.min(u))

    print(x.shape,y.shape,u.shape,v.shape)
    # fig,ax=plt.subplots(1,2)
    # ax[0].imshow(u,cmap='gray')
    # ax[1].imshow(v,cmap='gray')
    # plt.show()

    temp='temp2'

    if os.path.exists(temp):
        img3D_u=np.load(os.path.join(temp,'u.npy'))
        img3D_v=np.load(os.path.join(temp,'v.npy'))
    else:
        try:
            shutil.rmtree(temp)
        except :
            pass
        os.mkdir(temp)
        img3D_u=Bos_pipeline.Filterbackporcessing(u)
        img3D_v=Bos_pipeline.Filterbackporcessing(v)

        np.save(os.path.join(temp,'u.npy'), img3D_u)
        np.save(os.path.join(temp,'v.npy'), img3D_v)

    print(img3D_u.shape,img3D_v.shape)

    Dx=img3D_u[:,50,:]
    Dy=img3D_v[:,50,:]
    print(np.max(Dx),np.min(Dx))


    fig,ax=plt.subplots(1,2)
    ax[0].imshow(Dx,cmap='gray')
    ax[1].imshow(Dy,cmap='gray')
    plt.show()

    x=Bos_pipeline.solve_rho(x,y,Dx,Dy)

    fig,ax=plt.subplots(1,1)
    ax.imshow(x,cmap='gray')
    plt.savefig('SD.jpg')
    plt.show()

    # Bos_pipeline.slice_show(img3D_u)
    # print(img3D.shape)




    #photon=cv2.resize(photon,(photon.shape[1]*5,photon.shape[0]*5))
    # cv2.imshow('p',photon)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #print(photon)

    # path_save_img='01'
    # img3D=np.zeros((photon.shape[0],photon.shape[1],photon.shape[1]))
    # imgframe=[]
    # if os.path.exists(path_save_img)!=True:
    #     os.mkdir(path_save_img)
    
    # for i in range(photon.shape[0]):
    #     line=photon[i,:]  # 取出一行
    #     line=np.expand_dims(line,axis=1)
    #     line=np.repeat(line,360,axis=1)

    #     img_back=FBPIRandonTransform(line,len(line[0]),'SL',show=False)
    #     img_back,ratio,bias=normalize(img_back)
    #     cv2.imwrite(os.path.join(path_save_img,'{0:03d}.png'.format(i)),img_back)
    #     imgframe.append(img_back)
    #     img3D[i,:,:]=img_back
 
    #     img_back=cv2.resize(img_back,(img_back.shape[1]*5,img_back.shape[0]*5))
    #     cv2.imshow('p',img_back)
    #     cv2.waitKey(1)
    # cv2.destroyAllWindows()

    # i=0
    # while i<photon.shape[-1]:
        
    #     img=img3D[:,:,i].astype(np.uint8)
    #     print(img.shape,img)
    #     img=cv2.resize(img,(img.shape[1]*5,img.shape[0]*5))
    #     cv2.imshow('rrr',img)
    #     cv2.waitKey(0)
    #     i+=1        

    # cv2.destroyAllWindows()

    # print(img3D)

    # delta_3D=img3D*ratio+bias


    # sitk_img = sitk.GetImageFromArray(img3D, isVector=False)
    # sitk_img.SetSpacing(ConstPixelSpacing)
    # sitk_img.SetOrigin(Origin)
    # print(sitk_img)
    # sitk.WriteImage(sitk_img, os.path.join(SaveRawDicom, "sample" + ".mhd"))
