from cv2 import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


'''
2021.11.27
'''
# dir_path=r'RL\rot'
# save_dir_path=r'RL\temp'
# name=os.listdir(dir_path)
# path_src=[os.path.join(dir_path,i) for i in name]
# path_save=[os.path.join(save_dir_path,i) for i in name]


# path=iter(zip(path_src,path_save))

# for src,tag in path:
#     img=cv2.imread(src,1)
#     img=cv2.resize(img,(128,128))
#     # cv2.imshow('pp',img)
#     # cv2.waitKey(0)
#     cv2.imwrite(tag,img)
#     print(src,tag)

'''
2021.11.28 全重建结果显示
'''
# temp=r'temp'

# img3D_n = np.load(os.path.join(temp, 'n_3D.npy'))
# img3D_rho = np.load(os.path.join(temp, 'rho_3D.npy'))

# print(img3D_n.shape)

# slice=img3D_n[:,:,40]

# # plt.imshow(slice,cmap='gray')
# # plt.show()

# fig = plt.figure()
# #ax1 = plt.axes(projection='3d')
# ax = fig.add_subplot(111,projection='3d')  #这种方法也可以画多个子图
# H,W=slice.shape
# X=np.arange(0,W)
# Y=np.arange(0,H)
# X,Y=np.meshgrid(X,Y)
# Z=slice
# ax.plot_surface(X,Y,Z,rstride = 5, cstride = 5,cmap='rainbow')
# # ax.contour(X,Y,Z,zdir='z', offset=-3,cmap="rainbow")  #生成z方向投影，投到x-y平面
# # ax.contour(X,Y,Z,zdir='x', offset=-6,cmap="rainbow")  #生成x方向投影，投到y-z平面
# # ax.contour(X,Y,Z,zdir='y', offset=6,cmap="rainbow")   #生成y方向投影，投到x-z平面
# plt.show()

'''
2021.11.29
'''
import scipy.io as scio
from scipy import ndimage
path='XCAT512.mat'
bos01 = scio.loadmat(path)['XCAT512']

plt.imshow(bos01,cmap='gray')
plt.show()