# coding=utf-8
import math
import os
import shutil

import numpy as np
import scipy.io as scio
import torch.autograd
import torch.nn as nn
import torch.utils.data.dataloader as Dataloader
import torch.utils.data.dataset as Dataset
from matplotlib import pyplot as plt
from scipy import ndimage
from torch import tensor
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
from models import generator2


# class generator(nn.Module):
#     def __init__(self,z_dim,output):
#         super(generator, self).__init__()

#         self.gen = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(z_dim, 2560), 
#             nn.Dropout(),
#             nn.ReLU(True),  
#             nn.Linear(2560, 5120),  
#             nn.ReLU(True),  
#             nn.Dropout(),
#             nn.Linear(5120, 10240),  
#             nn.ReLU(True),  
            
#             nn.Linear(10240, 25000),  
#             nn.ReLU(True),  

#             nn.Linear(25000, 10240),  
#             nn.ReLU(True),  

#             nn.Linear(10240, output),  
#             nn.Tanh()
#         )
#         self.steps=steps
#         self.batchSize=batchSize
#     def forward(self, x):
#         x = self.gen(x)
#         x=x.view(batchSize,1,H,W)
#         output=DiscreteRadonTransform(x,self.steps,self.batchSize)
#         return x,output.view(self.batchSize,1,-1)

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
   
    channels = image.shape[-1]
    res = torch.zeros((channels, viewnum))
    if torch.cuda.is_available():
        res = res.cuda()
    for s in range(viewnum):
        angle = -math.pi - 180/viewnum*(s+1) * math.pi / 180
        A = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
        theta = np.array([[A[0, 0], A[0, 1], 0], [A[1, 0], A[1, 1], 0]])
        theta = torch.from_numpy(theta).type(torch.FloatTensor)
        theta = theta.unsqueeze(0)
        if torch.cuda.is_available():
            theta = theta.cuda()
        #image_temp = torch.from_numpy(image).type(torch.FloatTensor)
        #image_temp = image.unsqueeze(1)
        if torch.cuda.is_available():
            image = image.cuda()
        
        theta = theta.repeat(batchSize,1,1)
        grid = F.affine_grid(theta, torch.Size((batchSize,1,512,512)))
        rotation = F.grid_sample(image, grid)
        rotation = torch.squeeze(rotation)
        res[:,s] = torch.sum(rotation,dim=0)

    return res

def iRandon_by_DBP(photon):
    P,C=photon.shape
    photon=np.expand_dims(photon,axis=1)
    processing_P=np.repeat(photon,repeats=P,axis=1)/P
    processing_img=np.zeros((C,P,P))

    for i in range(C):
        processing_img[i,:,:]=ndimage.rotate(processing_P[:,:,i],-i*(180/steps),reshape=False)
    processing_img=np.average(processing_img,axis=0)
    return processing_img

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


if __name__ == '__main__':
    steps=5
    batchSize=1

    try:
        shutil.rmtree('figtemp')
    except :
        pass
    try:
        os.mkdir('figtemp')
    except:
        pass

    
    path=r'processing_data\XCAT512.mat'
    src_img = scio.loadmat(path)['XCAT512']
    H,W=src_img.shape

    photon_comp,field=Randon_Transformer(src_img,steps)

    photon_mask=photon_comp.copy()
    photon_mask[photon_comp>0.01]=1
    pH,pW=photon_mask.shape
    photon_mask=photon_mask[(pH//2-H//2):(pH//2+H//2),:]
    photon_mask=torch.from_numpy(photon_mask).type(torch.FloatTensor)
    photon_mask=photon_mask.view(batchSize,1,-1)


    # fig,ax=plt.subplots(1,2)
    # ax[0].imshow(photon_comp,cmap='gray')
    # ax[1].imshow(photon_mask,cmap='gray')

    # plt.show()
    #plt.close()

    f_H,f_W=field.shape
    processing_img_cmp=iRandon_by_DBP(photon_comp)
    processing_img_cmp=processing_img_cmp[(f_H//2-H//2):(f_H//2+H//2),(f_W//2-W//2):(f_W//2+W//2)]
    
    processing_img_cmp=np.expand_dims(processing_img_cmp,axis=0)
    processing_img_cmp=np.expand_dims(processing_img_cmp,axis=0)
    processing_img_cmp = torch.from_numpy(processing_img_cmp).type(torch.FloatTensor)
    #processing_img_cmp=processing_img_cmp.unsqueeze(1)
    processing_img_cmp=processing_img_cmp.view(batchSize,1,H,W)

    
    src_img=np.expand_dims(src_img,axis=0)
    src_img=np.expand_dims(src_img,axis=0)
    src_img = torch.from_numpy(src_img).type(torch.FloatTensor)
    photon=DiscreteRadonTransform(src_img,steps,batchSize)
    photon=photon.view(batchSize,1,-1)


    # photon=np.transpose(photon,(1,0))
    
    # photon=photon.reshape(1,-1)
    # photon=torch.from_numpy(photon).float()

    G = generator2(steps=steps,batchSize=1)
    if torch.cuda.is_available():
        G = G.cuda()

    criterion = nn.MSELoss()  # 是单目标二分类交叉熵函数
    crrr=nn.CrossEntropyLoss()
    #cri2=nn.MSELoss()
    #C222=TVLoss()

    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.001)

    if torch.cuda.is_available():
        src_img = src_img.cuda()
        processing_img_cmp=processing_img_cmp.cuda()
        photon_mask=photon_mask.cuda()

    iternum=0

    while True:
        iternum+=1
        processing_img,output,output2=G(processing_img_cmp)


        g_loss = criterion(output, photon)

        c_loss=crrr(output2,photon_mask)
        Gloss=g_loss+2*c_loss

        if Gloss<1e-10:
            break

        g_optimizer.zero_grad()  # 梯度归0
        Gloss.backward()  # 进行反向传播
        g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数

        if iternum%500==0:
            fig,ax=plt.subplots(2,3)
            ax[0][0].imshow(processing_img[0].cpu().data.squeeze(0).numpy(),cmap='gray')
            ax[0][1].imshow(output.view(batchSize,W,steps).cpu().data.squeeze(0).numpy(),cmap='gray')
            ax[0][2].imshow(output2.view(batchSize,W,steps).cpu().data.squeeze(0).numpy(),cmap='gray')
            ax[1][0].imshow(processing_img_cmp[0].cpu().data.squeeze(0).numpy(),cmap='gray')
            ax[1][1].imshow(photon.view(batchSize,W,steps).cpu().data.squeeze(0).numpy(),cmap='gray')
            ax[1][2].imshow(photon_mask.view(batchSize,W,steps).cpu().data.squeeze(0).numpy(),cmap='gray')
            plt.savefig('figtemp/iter{}.jpg'.format(iternum))
            plt.close()
        print('epoch{},G_loss{}'.format(iternum,g_loss))
