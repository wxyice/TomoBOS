"""
# Date: 2021.10.14
# Author: Wxyice
# Function：
#   
"""

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from mpl_toolkits.mplot3d import Axes3D
from openpiv import filters, pyprocess, scaling, tools, validation
from tqdm import tqdm

from DBP import LBPIRandonTransform
from Poissonsolve2D import buildMatrix
from solve import CG, SD, Gauss_Seidel
import timeit

class BOS():
    """
        BOS class
        to process BOS image
        Input:
        :Param: ZD
        :Param: ZA
        :Param: f
        :Param: n0
    """

    def __init__(self, ZD, ZA, f, n0) -> None:
        # constant
        self.G = 2.2244 * 1e-4

        self.ZD = ZD        # m
        self.ZA = ZA        # m
        self.f = f          # m
        self.n0 = n0        
        self.rho0=1.293     # kg/m^3
        #self.C_rho =(self.n0/self.G)*((self.ZD+self.ZA-f)/(self.ZD*f))
        self.C_n = (self.n0) * ((self.ZD + self.ZA - f) / (self.ZD * f))

    def cross_correlation(self,
                          path_a,
                          path_b,
                          winsize=32,
                          searchsize=38,
                          overlap=16,
                          dt=1,
                          show=False):
        """
        计算图片互相关算法，计算a，b两张图，a为前，b为后
        :param: winsize     # pixels, interrogation window size in frame A
        :param: searchsize  # pixels, search area size in frame B
        :param: overlap     # pixels, 50% overlap
        :param: dt          # sec, time interval between the two frames
        """
        a = tools.imread(path_a)
        b = tools.imread(path_b)

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
            u0,
            v0,
            sig2noise,
            threshold=1.05,
        )
        u2, v2 = filters.replace_outliers(
            u1,
            v1,
            method='localmean',
            max_iter=3,
            kernel_size=3,
        )
        # convert x,y to mm
        # convert u,v to mm/sec
        x, y, u3, v3 = scaling.uniform(
            x,
            y,
            u2,
            v2,
            scaling_factor=1056 / 1000,  # 96.52 pixels/millimeter  # 标定因子
        )
        # 0,0 shall be bottom left, positive rotation rate is counterclockwise
        x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)
        if show:
            fig, ax = plt.subplots(figsize=(8, 8))
            tools.save(x, y, u3, v3, mask, 'exp1_001.txt')
            tools.display_vector_field(
                'exp1_001.txt',
                ax=ax,
                scaling_factor=1056 / 100,
                scale=8,  # scale defines here the arrow length
                width=0.0035,  # width is the thickness of the arrow
                on_img=True,  # overlay on the image
                image_name=path_a,
            )
        return x, y, u3, v3

    def Filterbackporcessing(self, photon, show=False, save_path=None):
        """
            基于轴对称假设的反投影方法
            photon 幻影图像 二维 shape=2
        """
        H, W = photon.shape
        img3D = np.zeros((H, W, W))

        for i in tqdm(range(H)):
            line = photon[i, :]  # 取出一行
            line = np.expand_dims(line, axis=1)
            line = np.repeat(line, 180, axis=1)

            img_back = LBPIRandonTransform(line,
                                           len(line[0]))  # ,'SL',show=False)
            img3D[i, :, :] = img_back
            if show:
                plt.ion()
                plt.imshow(img_back, cmap='gray')
                plt.pause(1)
                plt.close()
        if save_path != None:
            np.save(save_path, img3D)
        return img3D

    def load_from_mat(self, path):
        bos01 = scio.loadmat(path)
        self.delta_magnitude = bos01['velocity_magnitude']
        self.delta_u, self.delta_v = bos01['u_original'], bos01['v_original']
        self.x, self.y = bos01['x'], bos01['y']

    def slice_show(self, img3D):
        i = 0
        while i < img3D.shape[-1]:
            img = img3D[:, :, i]
            print(img.shape, img)
            plt.ion()
            plt.imshow(img, cmap='gray')
            plt.pause(1)
            plt.close()
            i += 1

    def solve_rho_for_slice(self, x, y, Dx, Dy, solver='CG'):
        """
        求解密度场
        :Param:
        """
        # x,y,Dx,Dy，xy是坐标位置矩阵，Dx，Dy应该来自于反演后的结果，所得到的B结果还需要一个系数比例C
        # 去除nan数据
        Dx[np.isnan(Dx)] = 0
        Dy[np.isnan(Dy)] = 0

        A_n, B_n = buildMatrix(x, y, Dx, Dy, self.C_n,boundary=self.n0)
        H, W = x.shape

        if solver == 'CG':            n = CG(A_n, B_n,maxiter=1000)
        elif solver == 'SD':          n = SD(A_n, B_n, maxiter=5000)
        elif solver == 'Gauss_Seidel':n = Gauss_Seidel(A_n, B_n)

        n = n.reshape((H, W))
        rho=(n-1)/self.G
        return n,rho
    
    def solve_rho_for_all(self,x,y,img3D_u,img3D_v):
        Z,X,Y=img3D_u.shape
        n_3D=np.zeros((Z,X,Y))
        rho_3D=np.zeros((Z,X,Y))

        for slice in tqdm(range(X)):
            Dx=img3D_u[:,slice,:]
            Dy=img3D_v[:,slice,:]
            n,rho=self.solve_rho_for_slice(x,y,Dx,Dy,solver='CG')
            n_3D[:,slice,:]=n
            rho_3D[:,slice,:]=rho

        return n_3D, rho_3D
        
    def show_xyuv(self,x,y,u,v,save_dir):
        fig, ax = plt.subplots(2, 2)
        ax[0][0].imshow(x, cmap='gray'), ax[0][0].set_title('x')
        ax[0][1].imshow(y, cmap='gray'), ax[0][1].set_title('y')
        ax[1][0].imshow(u, cmap='gray'), ax[1][0].set_title('u')
        ax[1][1].imshow(v, cmap='gray'), ax[1][1].set_title('v')
        plt.savefig(os.path.join(save_dir, 'xyuv.jpg'))
        plt.close()

    def show_slicexy(self,sx,sy,save_dir):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(sx, cmap='gray')
        ax[1].imshow(sy, cmap='gray')
        plt.savefig(os.path.join(save_dir, 'sxsy.jpg'))
        plt.close()

    def show_line_plt(self,img,save_path=None):
        H,W=img.shape
        X=np.array([i for i in range(W)])
        line=50
        Y=img[line,:]

        fig,ax=plt.subplots(1,1)

        ax.plot(X,Y)
        # for line in range(H):
        #     Y=img[line]+line/2e8
        #     ax.plot(X,Y)

        if save_path!=None:
            plt.savefig(os.path.join(save_path,'line.jpg'))
            plt.close()
        else:
            plt.show()
            plt.close()
        print('done')

    def show_slice_in_3D(self,img):
        fig = plt.figure()
        #ax1 = plt.axes(projection='3d')
        ax = fig.add_subplot(111,projection='3d')  #这种方法也可以画多个子图
        H,W=img.shape
        X=np.arange(0,W)
        Y=np.arange(0,H)
        X,Y=np.meshgrid(X,Y)
        Z=img
        ax.plot_surface(X,Y,Z,rstride = 1, cstride = 1,cmap='rainbow')
        ax.contour(X,Y,Z,zdir='z', offset=-3,cmap="rainbow")  #生成z方向投影，投到x-y平面
        ax.contour(X,Y,Z,zdir='x', offset=-6,cmap="rainbow")  #生成x方向投影，投到y-z平面
        ax.contour(X,Y,Z,zdir='y', offset=6,cmap="rainbow")   #生成y方向投影，投到x-z平面
        plt.show()


if __name__ == '__main__':
    
    ZD = 0.5
    ZA = 0.5
    f = 0.05
    n0 = 1.00029
    rho0=1.293

    # set the PIVLab data path
    mat_path = r'processing_data\01_PIVprocess.mat'
    mat_path = r'processing_data\PIVlab_try2.mat'

    # for initialize the test file
    import datetime
    debug = 'test'
    if os.path.exists(debug)!=True:
        os.mkdir(debug)

    now_time = str(datetime.datetime.now().strftime('%Y%m%d %H%M%S'))
    path_for_debug = os.path.join(debug, now_time)
    path_for_3Ddxdy=os.path.join(debug,now_time,'3D')
    path_for_plt=os.path.join(debug,now_time,'plt')
    path_for_result=os.path.join(debug,now_time,'result')
    
    os.mkdir(os.path.join(path_for_debug))
    os.mkdir(path_for_3Ddxdy)
    os.mkdir(path_for_plt)
    os.mkdir(path_for_result)

    # initialize the BOS class
    Bos_pipeline = BOS(ZD, ZA, f, n0)

    # PIVLab + LBP
    Bos_pipeline.load_from_mat(mat_path)
    photon = Bos_pipeline.delta_magnitude[0][0]
    x = Bos_pipeline.x[0][0]
    y = Bos_pipeline.y[0][0]
    u = Bos_pipeline.delta_u[0][0]
    v = Bos_pipeline.delta_v[0][0]
    
    # plt.hist(u.reshape(-1,1),bins=100)
    # plt.show()
    # plt.hist(v.reshape(-1,1),bins=100)
    # plt.show()
    
    # openPIV + LBP
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

    # print(x.shape, y.shape, u.shape, v.shape)
    # fig,ax=plt.subplots(1,2)
    # ax[0].imshow(u,cmap='gray')
    # ax[1].imshow(v,cmap='gray')
    # plt.show()

    temp = 'temp'#os.path.join(debug_path,'temp')

    if os.path.exists(temp):
        img3D_u = np.load(os.path.join(temp, 'u.npy'))
        img3D_v = np.load(os.path.join(temp, 'v.npy'))
        np.save(os.path.join(path_for_3Ddxdy,'u.npy'),img3D_u)
        np.save(os.path.join(path_for_3Ddxdy,'v.npy'),img3D_v)
    else:
        try:
            shutil.rmtree(temp)
        except:
            pass
        os.mkdir(temp)
        img3D_u = Bos_pipeline.Filterbackporcessing(u)
        img3D_v = Bos_pipeline.Filterbackporcessing(v)

        np.save(os.path.join(temp, 'u.npy'), img3D_u)
        np.save(os.path.join(temp, 'v.npy'), img3D_v)

    # print(img3D_u.shape, img3D_v.shape)

    # img3D 的格式[z,x,y] 获取切片
    Dx = img3D_u[:, :, 50]
    Dy = img3D_v[:, :, 50]


    start=timeit.default_timer()
    n,rho = Bos_pipeline.solve_rho_for_slice(x, y, Dx, Dy)
    end=timeit.default_timer()

    np.save(os.path.join(path_for_result,'n.npy'),n)
    np.save(os.path.join(path_for_result,'rho.npy'),rho)

    # print(end-start)

    # n_3D, rho_3D=Bos_pipeline.solve_rho_for_all(x,y,img3D_u,img3D_v)
    
    # np.save(os.path.join(path_for_result,'n_3D.npy'),n_3D)
    # np.save(os.path.join(path_for_result,'rho_3D.npy'),rho_3D)



    # Bos_pipeline.show_xyuv(x,y,u,v,path_for_plt)
    # Bos_pipeline.show_slicexy(Dx,Dy,path_for_plt)
    # Bos_pipeline.show_line_plt(n)

    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(n, cmap='gray')
    # ax[1].imshow(rho,cmap='gray')
    # ax[0].set_title('n')
    # ax[1].set_title('rho')
    # plt.savefig(os.path.join(path_for_plt, 'n_rho.jpg'))
    # plt.show()



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
