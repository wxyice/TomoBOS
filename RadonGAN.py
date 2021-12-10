# coding=utf-8
import math
import os
import shutil

import numpy as np
import scipy.io as scio
import torch
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
    
    channels = image.shape[-1]
    B,C,H,W=image.shape
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
        #grid = F.affine_grid(theta, torch.Size((batchSize,1,512,512)))
        grid = F.affine_grid(theta, torch.Size((batchSize,1,H,W)))
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


def normalization(img):
    return (img-np.min(img))/(np.max(img)-np.min(img))

if __name__ == '__main__':
    import datetime

    from cv2 import cv2

    from modellayer import UNet

    steps=5
    batchSize=1

    resize_size=512   # 使用全连接最多到128，512内存会过大


    root='test/ART_test'
    
    now_time = str(datetime.datetime.now().strftime('%Y%m%d %H%M%S'))+'-'+str(steps)+'-'+str(resize_size)
    try:
        os.mkdir(root)
    except:
        pass
    os.mkdir(os.path.join(root,now_time))   
    save_path=os.path.join(root,now_time)
    
    path=r'processing_data\XCAT512.mat'
    src_img = scio.loadmat(path)['XCAT512']
    H,W=src_img.shape
    
    if src_img.shape!=(resize_size,resize_size):
        src_img=cv2.resize(src_img,(resize_size,resize_size))

    src_img=normalization(src_img)

    src_img=np.expand_dims(src_img,axis=0)
    src_img=np.expand_dims(src_img,axis=0)
    src_img = torch.from_numpy(src_img).type(torch.FloatTensor)
    photon=DiscreteRadonTransform(src_img,steps,batchSize)
    
    photon=photon.cpu().numpy()

    # photon=cv2.resize(photon,(10,W))

    photon_mask=photon.copy()#.cpu().numpy()

    photon_mask[photon_mask>0.01]=1

    photon=torch.from_numpy(photon).type(torch.FloatTensor)
    photon_mask = torch.from_numpy(photon_mask).type(torch.FloatTensor)

    photon=photon.view(batchSize,1,-1)
    photon_mask=photon_mask.view(batchSize,1,-1)

    # steps=10
    G = UNet(1,1,steps,batchSize)

    if torch.cuda.is_available():
        G = G.cuda()

    MSELoss = nn.MSELoss()  
    CELoss=nn.CrossEntropyLoss()


    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.001)

    if torch.cuda.is_available():
        src_img = src_img.cuda()
        src=torch.rand((batchSize,1,H,W)).cuda()
        photon=photon.cuda()
        photon_mask=photon_mask.cuda()

    iternum=0
    
    while True:
        iternum+=1
        processing_img,output,output2=G(src)
        g_loss = MSELoss(output, photon)
        c_loss=CELoss(output2,photon_mask)

        Gloss=g_loss+c_loss

        if Gloss<1e-10:
            break

        g_optimizer.zero_grad()  
        Gloss.backward()  
        g_optimizer.step() 

        err=(processing_img[0].cpu().data.squeeze(0).numpy()-src_img[0].cpu().data.squeeze(0).numpy())
        err_photon=(output.view(batchSize,W,steps).cpu().data.squeeze(0).numpy()-photon.view(batchSize,W,steps).cpu().data.squeeze(0).numpy())

        err_mask=(output2.view(batchSize,W,steps).cpu().data.squeeze(0).numpy()-photon_mask.view(batchSize,W,steps).cpu().data.squeeze(0).numpy())
        
        if iternum%100==0:
            fig,ax=plt.subplots(3,3)
            ax[0][0].imshow(processing_img[0].cpu().data.squeeze(0).numpy())#,cmap='gray')
            ax[0][1].imshow(output.view(batchSize,W,steps).cpu().data.squeeze(0).numpy())#,cmap='gray')
            ax[0][2].imshow(output2.view(batchSize,W,steps).cpu().data.squeeze(0).numpy())#,cmap='gray')
            
            # ax[1][0].imshow(processing_img_cmp[0].cpu().data.squeeze(0).numpy())#,cmap='gray')
            ax[1][0].imshow(src_img[0].cpu().data.squeeze(0).numpy())#,cmap='gray')
            ax[1][1].imshow(photon.view(batchSize,W,steps).cpu().data.squeeze(0).numpy())#,cmap='gray')
            ax[1][2].imshow(photon_mask.view(batchSize,W,steps).cpu().data.squeeze(0).numpy())#,cmap='gray')

            ax[2][0].imshow(err)#,cmap='gray')
            ax[2][1].imshow(err_photon)#,cmap='gray')
            ax[2][2].imshow(err_mask)

            cv2.imwrite(os.path.join(save_path,'CViter{}Gloss{}err{}.jpg'.format(iternum,Gloss,np.sum(err**2))),((processing_img[0].cpu().data.squeeze(0).numpy())*255).astype(np.uint8))
            plt.savefig(os.path.join(save_path,'iter{}Gloss{}err{}.jpg'.format(iternum,Gloss,np.sum(err**2))))
            plt.close('all')
        
        print('epoch{},G_loss{} err {}'.format(iternum,g_loss,np.sum(err**2)))
