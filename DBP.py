#----------------------------------------------
# Date: 2021.10.14
# Author: Wxyice
# Funtion：
#   实现滤波反投影方法
#----------------------------------------------


import os
import time

import imageio
import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2
from icecream import ic
from scipy import ndimage
from scipy.signal import convolve


#两种滤波器的实现
def RLFilter(N, d):
    filterRL = np.zeros((N,))
    for i in range(N):
        filterRL[i] = - 1.0 / np.power((i - N / 2) * np.pi * d, 2.0)
        if np.mod(i - N / 2, 2) == 0:
            filterRL[i] = 0
    filterRL[int(N/2)] = 1 / (4 * np.power(d, 2.0))
    return filterRL

def SLFilter(N, d):
    filterSL = np.zeros((N,))
    for i in range(N):
        #filterSL[i] = - 2 / (np.power(np.pi, 2.0) * np.power(d, 2.0) * (np.power((4 * (i - N / 2)), 2.0) - 1))
        filterSL[i] = - 2 / (np.pi**2.0 * d**2.0 * (4 * (i - N / 2)**2.0 - 1))
    return filterSL

# def RLfilter_Omega(N):
#     # RL 滤波器在频率域表示为斜坡滤波
#     filterRL=np.zeros((N,))
#     for i in range(N):
#         filterRL[i] = abs((i-N/2))/(N/2)
#     return filterRL

def FBPIRandonTransform(image, steps, filter=None, show=True):
    #show=True
    #定义用于存储重建后的图像的数组
    channels = image.shape[0]
    #print('ccccc'+str(channels))
    origin = np.zeros((steps, channels, channels))
    #print(origin.shape)
    if filter is None:
        Filter = RLFilter(channels, 1)
        #filter = SLFilter(channels, 1)
    elif filter=='RL':
        Filter = RLFilter(channels,1)
    elif filter =='SL':
        Filter = SLFilter(channels,1) 
    #filter=RLfilter_Omega(channels)
    for i in range(0,steps):
        projectionValue = image[:, i]
        projectionValueFiltered = convolve(Filter, projectionValue, "same")
        projectionValueExpandDim = np.expand_dims(projectionValueFiltered, axis=0)
        projectionValueRepeat = projectionValueExpandDim.repeat(channels, axis=0)
        
        ic(projectionValueFiltered)
        ic(projectionValueRepeat)

        origin[i] = ndimage.rotate(projectionValueRepeat, i*360/steps, reshape=False).astype(np.float64)

        projectionValueRepeat=normalize_image(projectionValueRepeat)
        b=normalize_image(np.sum(origin[:i], axis=0)/i)

        if show:
            cv2.imshow('rrr',projectionValueRepeat.astype(np.uint8))
            cv2.imshow('ttt',b.astype(np.uint8))
            
            if os.path.exists(filter)!=True:
                os.mkdir(filter)
                os.mkdir(os.path.join(filter,'zhong'))
                os.mkdir(os.path.join(filter,'rot'))
                os.mkdir(os.path.join(filter,'singel'))
            
            cv2.imwrite(os.path.join(filter,'zhong',"{1}{0:03d}.jpg".format(i,filter)),(b.astype(np.uint8)))
            cv2.imwrite(os.path.join(filter,'rot',"{1}{0:03d}.jpg".format(i,filter)),normalize_image(origin[i]).astype(np.uint8))
            cv2.imwrite(os.path.join(filter,'singel',"{1}{0:03d}.jpg".format(i,filter)),projectionValueRepeat.astype(np.uint8))
            cv2.waitKey(10)
                    
    iradon = np.sum(origin, axis=0)
    return iradon


def LBPIRandonTransform(image, steps):
    # 直接反投影算法
    #定义用于存储重建后的图像的数组
    channels = len(image[0])
    print(channels)
    origin = np.zeros((steps, channels, channels))
    for i in range(steps):
    	#传入的图像中的每一列都对应于一个角度的投影值
    	#这里用的图像是上篇博文里得到的Radon变换后的图像裁剪后得到的
        projectionValue = image[:, i]
        ic(projectionValue)

        #这里利用维度扩展和重复投影值数组来模拟反向均匀回抹过程
        projectionValueExpandDim = np.expand_dims(projectionValue, axis=0)
        projectionValueRepeat = projectionValueExpandDim.repeat(channels, axis=0)

        origin[i] = ndimage.rotate(projectionValueRepeat, i*180/steps, reshape=False)
        origin[i]=origin[i].astype(np.float64)


        cv2.imshow('rrr',projectionValueRepeat)
        cv2.imshow('ttt',(np.sum(origin[:i], axis=0)/i).astype(np.uint8))
        cv2.imwrite(os.path.join('LBP','zhong',"LBP{0:03d}.jpg".format(i)),(np.sum(origin[:i], axis=0)/i).astype(np.uint8))
        cv2.imwrite(os.path.join('LBP','rot',"LBP{0:03d}.jpg".format(i)),origin[i].astype(np.uint8))
        cv2.imwrite(os.path.join('LBP','singel',"LBP{0:03d}.jpg".format(i)),projectionValueRepeat)
        cv2.waitKey(3)

    iradon = np.sum(origin, axis=0)
    return iradon

def normalize_image(image):
    # for opencv
    image-=np.min(image)
    image=(image/np.max(image)*255).astype(np.uint8)
    return image

def show_image(image,iradon):
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(iradon, cmap='gray')
    
    plt.show()

if __name__ == '__main__':
    ic.disable()

    image = cv2.imread("radonshepplogan.png", cv2.IMREAD_GRAYSCALE)

    channels=len(image[0])
    #iradon_LBP= LBPIRandonTransform(image, len(image[0]))
    iradon_RF = FBPIRandonTransform(image, len(image[0]), 'RL')#Filter(channels, 1))
    iradon_SF = FBPIRandonTransform(image, len(image[0]), 'SL')#Filter(channels, 1))
    #print(iradon_SF)

    
    #show_image(image,iradon_LBP)
    #cv2.imwrite("LBP.png",iradon_LBP)
    show_image(image,iradon_RF)
    show_image(image,iradon_SF)

    #cv2.imwrite('SF.png',iradon_SF)

