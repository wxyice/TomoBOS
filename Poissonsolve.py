import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import scipy.io as scio
import SimpleITK as sitk
from cv2 import cv2
from icecream import ic
from numpy.core.numeric import Inf
from numpy.lib.financial import npv
from PIL import Image

from DBP import FBPIRandonTransform
from show3D import plot_3d


class BOS():
    def __init__(self) -> None:
        pass
    def loadmat(self,path):
        mat_file=scio.loadmat(path)
        Displ=mat_file["Displacement_POisson"][0][0]
        self.x=Displ[0]
        self.y=Displ[1]
        self.u=Displ[2]
        self.v=Displ[3]

def create_RHS(Displ):# RHS,Nx,Ny
    '''
    This function creates the Rhs of the Poisson's equation. 
    The Rhs is obtained summing dx+dy (defined as Displ.u and Displ.v).
    '''
    u=Displ.u
    w=Displ.v
    x=Displ.x
    z=Displ.y

    width,height=u.shape
    # Preallocation for the variables du and dw;
    du=np.zeros((width-1,height-1),dtype=np.float64)
    dw=np.zeros((width-1,height-1),dtype=np.float64)

    # Compute the central differences 
    for k in range(1,width-1):
        for j in range(1,height-1):  
            du[k][j]=(u[k+1][j]-u[k-1][j])/2*(abs(x[1][1]-x[1][0]))
        
            dw[k][j]=(w[k][j+1]-w[k][j-1])/2*(abs(z[0][1]-z[0][0]))

    Rhs=du+dw
    Rhs[Rhs==np.Infinity] = 0 

    Nz,Nx=Rhs.shape
    return Rhs,Nz,Nx

def crop_field(Displacement_POisson,Lx,Lz):
    Displ=Displacement_POisson
    Minimum=np.min(Displ.y)
    Displ.y=Displ.y-abs(Minimum)
    #Magnitude=np.sqrt(Displ.u**2+Displ.v**2)
    a,b=Displ.x.shape

    Dxx=Lx/a
    Dyy=Lz/b
    nx_pixels_crop=700#  %250
    ny_pixels_crop=350#  %200
    Lx=Lx-nx_pixels_crop
    Lz=Lz-ny_pixels_crop
    Dx_pixels=round(nx_pixels_crop/Dxx)#  % number of pixels to crop in th eright and left side of the image
    Dy_pixels=round(ny_pixels_crop/Dyy)#  % number of pixels to crop in th eright and left side of the image
    Displ.x=Displ.x[Dx_pixels-1:-Dx_pixels,Dy_pixels-1:-Dy_pixels]
    Displ.y=Displ.y[Dx_pixels-1:-Dx_pixels,Dy_pixels-1:-Dy_pixels]
    Displ.u=Displ.u[Dx_pixels-1:-Dx_pixels,Dy_pixels-1:-Dy_pixels]
    Displ.v=Displ.v[Dx_pixels-1:-Dx_pixels,Dy_pixels-1:-Dy_pixels]
    #Magnitude_crop=sqrt(Displ.u.^2+Displ.v.^2)

    return Displ

def scaledata(datain,minval,maxval):
    '''
    % Program to scale the values of a matrix from a user specified minimum to a user specified maximum
    %
    % Usage:
    % outputData = scaleData(inputData,minVal,maxVal);
    %
    % Example:
    % a = [1 2 3 4 5];
    % a_out = scaledata(a,0,1);
    % 
    % Output obtained: 
    %            0    0.1111    0.2222    0.3333    0.4444
    %       0.5556    0.6667    0.7778    0.8889    1.0000
    %
    % Program written by:
    % Aniruddha Kembhavi, July 11, 2007
    '''

    dataout = datain - np.min(datain)
    dataout = (dataout/np.max(datain))*(maxval-minval)
    dataout = dataout + minval
    return dataout
#---------load Displacement_POisson.mat----------------
#load Displacement_POisson1.mat
Lx=1720
Lz=2304

Displ=BOS()
Displ.loadmat(path='Displacement_POisson1.mat')


# #print(Displ.x)
# print(Displ.x.shape)

Displ=crop_field(Displ,Lx,Lz)
print(Displ.x.shape)
Rhs,_,_=create_RHS(Displ)

# print(Rhs.shape)
# print(Rhs)

Nx,Nz=Rhs.shape

# [Nx,Nz]=size(Rhs)

#Specifying parameters (check why It does not work if we change them)

niter=100000                      # Number of iterations 
dx=Lx/(Nx-1)                      # Width of space step(x)
dy=Lz/(Nz-1)                      # Width of space step(y)
x=np.arange(0,Lx,dx)
y=np.arange(0,Lz,dy)
#x=np.linspace(0,Lx,dx)               # Range of x(0,2) and specifying the grid points
#y=np.linspace(0,Lx,dx)                       # Range of y(0,2) and specifying the grid points
b=np.zeros((Nx,Nz))                    # Preallocating b
pn=np.zeros((Nx,Nz))                   # Preallocating pn


#Initial Conditions

p=np.zeros((Nx,Nz))                  #Preallocating p

 
#Rhs=Const.*fliplr(Rhs);   

#b=(Rhs)/max(max(Rhs));%b=(Rhs')/norm(Rhs');
b=Rhs



# i=np.arange(1,Nx-2,1)
# j=np.arange(1,Nz-2,1)

# Poisson equation solution (iterative) method

tol = 1e-4			 # error is 1%
maxerr = 1000	     # initial error
iter = 0
pn=p

while maxerr > tol:
    iter = iter + 1
    print('Iteration no:{0} '.format(iter))
    for i in range(1,Nx-1):
        for j in range(1,Nz-1):
            p[i][j]=((dy**2*(pn[i+1][j]+pn[i-1][j]))+(dx**2*(pn[i][j+1]+pn[i][j-1]))-(b[i][j]*dx**2*dy*2))/(2*(dx**2+dy**2))
    #Boundary conditions 
    
    # Neumann's conditions   % dp/dx|end=dp/dx|end-1 
    p[0,:]=p[1,:]
    p[-1,:]=p[-2,:]
    
    # # Neumann's conditions
    # p(:,1)=p(:,2);
    # p(:,end)=p(:,end-1);

    maxerr =np.max(np.abs((p-pn)))
    print('Maximum error is {}'.format(maxerr))
    pn=p


# Plotting the solution

PG2_gray=p*255
n_max=1.43   
n_min=1.332
n2= scaledata(PG2_gray,n_min,n_max)


#figure：新的画布
fig=plt.figure()
#axes：坐标轴
ax1=plt.axes(projection='3d')
#ax=fig.add_subplot(111,projection='3d')#画子图

plt.subplot(121)
h=ax1.plot_surface(x,y,p[1:,1:])    

ax1.xlabel('Spatial co-ordinate (x) \rightarrow')
ax1.ylabel('{\leftarrow} Spatial co-ordinate (y)')
ax1.zlabel('Solution profile (P) \rightarrow')

plt.subplot(122)
h=ax1.plot_surface(x,y,n2);       

ax1.xlabel('Spatial co-ordinate (x) \rightarrow')
ax1.ylabel('{\leftarrow} Spatial co-ordinate (y)')
ax1.zlabel('Solution profile (P) \rightarrow')
plt.show()

# plot(n2(5,:),y)



