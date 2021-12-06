# coding=utf-8
from torch import tensor
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import os
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as Dataloader
from cv2 import cv2 
import shutil

import numpy as np

import os,shutil
import scipy.io as scio
from matplotlib import pyplot as plt
from scipy import ndimage
from tqdm import tqdm
import math
from torch.nn import functional as F

class generator(nn.Module):
    def __init__(self,z_dim,output):
        super(generator, self).__init__()

        self.gen = nn.Sequential(
            nn.Linear(z_dim, 2560), 
            nn.ReLU(True),  
            nn.Linear(2560, 1024),  
            nn.ReLU(True),  
            nn.Linear(1024, 1024),  
            nn.ReLU(True),  
            nn.Linear(1024, output),  
            nn.Tanh()
        )
    def forward(self, x):
        x = self.gen(x)
        return x

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

def DiscreteRadonTransform(image, viewnum, batchSize):
    # image: batchSize*imgSize*imgSize
    channels = len(image[0])
    res = torch.zeros((channels, viewnum))
    #res = res.cuda()
    for s in range(viewnum):
        angle = -math.pi - 180/viewnum*(s+1) * math.pi / 180
        A = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
        theta = np.array([[A[0, 0], A[0, 1], 0], [A[1, 0], A[1, 1], 0]])
        theta = torch.from_numpy(theta).type(torch.FloatTensor)
        theta = theta.unsqueeze(0)

        #theta = theta.cuda()
        image_temp = torch.from_numpy(image).type(torch.FloatTensor)
        image_temp = image_temp.unsqueeze(1)
        #image_temp = image_temp.cuda()
        
        theta = theta.repeat(batchSize,1,1)
        grid = F.affine_grid(theta, torch.Size((batchSize,1,512,512)))
        rotation = F.grid_sample(image_temp, grid)
        rotation = torch.squeeze(rotation)
        res[:,s] = torch.sum(rotation,dim=0)

    return res

path=r'processing_data\XCAT512.mat'
src_img = scio.loadmat(path)['XCAT512']
src_img = np.expand_dims(src_img,axis=0)
res=DiscreteRadonTransform(src_img,180,1)

plt.imshow(res.numpy(),cmap='gray')
plt.show()

