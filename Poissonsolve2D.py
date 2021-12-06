"""
Date: 2021.11.18
Author: Wxyice

solve the Poisson Function in 2D, 

用有限差分的方法求解2维泊松方程
"""
import numpy as np
import os
from matplotlib import pyplot as plt
import numba


@numba.jit(nopython=True,cache=True)
def grad(x,y,Dx,Dy):
    """
    x,y 是各个控制点的对应真实位置，分别用xy坐标表示，应该来源于标定函数
    u,v 是各个控制点的位移量
    这个函数是实现求导功能
    """    
    H,W=Dx.shape

    G=np.zeros((H,W))
    # 循环求导
    for i in range(1,H-1):
        for j in range(1,W-1):
            Gx=(Dx[i][j+1]-Dx[i][j-1])/(2*(x[i][j+1]-x[i][j]))
            Gy=(Dy[i+1][j]-Dy[i-1][j])/(2*(y[i+1][j]-y[i][j]))
            G[i][j]=Gx+Gy
    # G.shape=W-2,H-2
    return G

def cut_domain(domain):
    return domain[1:-1,1:-1].copy()


@numba.jit(nopython=True,cache=True)
def buildMatrix(x,y,Dx,Dy,C,boundary):
    G=grad(x,y,Dx,Dy)
    # plt.imshow(G,cmap='gray')
    # plt.show()
    n0=boundary
    H,W=Dx.shape
    num=W*H                 # 一共有num个元素，依次按照行编号，每个点i,j的编号为 i*W+j
    A=np.zeros((num,num))
    B=np.zeros((num))
    dx=abs(x[0][2]-x[0][1])
    dy=abs(y[1][0]-y[0][0])

    ay=-dy**2/(2*(dy**2+dx**2))
    ax=-dx**2/(2*(dy**2+dx**2))

    ing=(dx**2*dy**2)/(2*(dy**2+dx**2))

    G=-C*G*(dx**2*dy**2)/(2*(dy**2+dx**2))
    for i in range(0,H):
        for j in range(0,W):
            index=i*W+j
            if i==0 or i==H-1 or j==0 or j==W-1: # 边界点
                A[index][index]=1
                B[index]=n0
            else:
                B[index]=G[i][j]    # 
                A[index][i*W+j]=1
                A[index][i*W+(j-1)],A[index][i*W+(j+1)]=ay,ay
                A[index][(i+1)*W+j],A[index][(i-1)*W+j]=ax,ax
    # for i in range(0,H):
    #     for j in range(0,W):
    #         index=i*W+j
    #         if i==0:
    #             B[index]=B[index+W]
    #         elif i==H-1:
    #             B[index]=B[index-W]
    #         elif j==0:
    #             B[index]=B[index+1] 
    #         elif j==W-1:
    #             B[index]=B[index-1]
    #         else:
    #             continue       
    return A,B
if __name__ == '__main__':
    """
    注意：x是竖直方向，y是水平方向
        
    """

    x=[
        [0,1,2,3,4,5,6],
        [0,1,2,3,4,5,6],
        [0,1,2,3,4,5,6],
        [0,1,2,3,4,5,6],
        [0,1,2,3,4,5,6]
    ]
    y=[
        [0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1], 
        [2,2,2,2,2,2,2],
        [3,3,3,3,3,3,3],
        [4,4,4,4,4,4,4]
    ]
    test_x=[
        [1,2,3,4,5,6,7], 
        [3,4,5,7,8,2,5], 
        [4,5,8,1,5,9,2], 
        [2,4,8,1,8,0,2], 
        [1,4,5,7,8,6,1]
    ]
    test_y=[
        [7,2,3,4,0,6,5], 
        [3,3,4,7,8,2,5], 
        [4,5,8,6,2,9,2], 
        [2,4,8,6,7,8,2], 
        [3,9,5,6,8,1,1]
    ]
    x=np.array(x)
    y=np.array(y)
    test_x=np.array(test_x)
    test_y=np.array(test_y)

    print(test_x.shape)
    G=grad(x,y,test_x,test_y)
    # print(G)
    B=G.reshape(-1)
    # print(B.shape)
    A,B=buildMatrix(x,y,test_x,test_y)
    # print(A)
    # print(B)
    # print(A.shape,B.shape)

    from solve import *
    import matplotlib.pyplot as plt

    x1=CG(A,B)
    x2=SD(A,B)
    x3=Gauss_Seidel(A,B)

    x1=cut_domain(x1.reshape(5,7))
    x2=cut_domain(x2.reshape(5,7))
    x3=cut_domain(x3.reshape(5,7))

    fig,ax=plt.subplots(3,1)
    ax[0].imshow(x1,cmap='gray')
    ax[1].imshow(x2,cmap='gray')
    ax[2].imshow(x3,cmap='gray')
    plt.show()

    # print(x1)
    # print(x2)
    # print(x3)



    
