import os

import imageio
import numpy as np
from cv2 import VideoCapture
from cv2 import cv2
from cv2 import cv2 as cv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import feature, measure


def plot_3d(image, threshold=20):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]
    #p=image
    verts,faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.savefig('frame1.jpg')

def make_gif(name,path):
    buff=[]
    filename=iter(sorted([os.path.join(path,i) for i in os.listdir(path)]))
    while True:
        try:
            img_path=next(filename)
            print(img_path)
            frame=cv.imread(img_path,1)
            buff.append(frame)
        except Exception as e:
            print(e)
            break
    gif=imageio.mimsave(name,buff,'GIF',duration=0.05)
       

if __name__ == '__main__':

    # if os.path.exists('01.npy'):
    #     img3D=np.load('01.npy')
    #     plot_3d(img3D,threshold=60)
    # else:
    #     path_dir='01'  
    #     img_name=sorted(os.listdir(path_dir))

    #     initimg=cv2.imread(os.path.join(path_dir,img_name[0]))


    #     img3D=np.zeros((len(img_name),initimg.shape[0],initimg.shape[1]))

    #     for i in range(len(img_name)):
    #         img=cv2.imread(os.path.join(path_dir, img_name[i]),0)
    #         img3D[i,:,:]=img
    #     np.save('01', img3D)
    #     plot_3d(img3D)    
    
    img3D=np.load('01.npy')
    xy=img3D[:,30,:]
    print(xy)
    xx = np.arange(0,104,1)
    yy = np.arange(0,101,1)
    X, Y = np.meshgrid(xx, yy)
    Z =xy# np.sin(X)+np.cos(Y)

    fig = plt.figure()  #定义新的三维坐标轴
    ax3 = plt.axes(projection='3d')

    #作图
    #ax3.plot_surface(X,Y,Z,cmap='rainbow')
    ax3.plot_surface(X,Y,Z,rstride = 1, cstride = 1,cmap='rainbow')
    #ax3.contour(X,Y,Z, zdim='z',offset=-2,cmap='rainbow')   #等高线图，要设置offset，为Z的最小值
    plt.show()
    # name='xy.gif'
    # # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # # out = cv2.VideoWriter('output.avi', fourcc, 10, (img3D.shape[1], img3D.shape[0]))
    # i=0
    # buff=[]
    # count=0
    # while i<img3D.shape[0]:
        
    #     img=img3D[i,:,:].astype(np.uint8)
    #     buff.append(img)
    #     print(img.shape,img)

    #     img=cv2.resize(img,(img.shape[1]*2,img.shape[0]*2))
    #     #out.write(img)
    #     cv2.imwrite(os.path.join('gif_src', "{0:05d}.jpg".format(count)),img)
    #     cv2.imshow('rrr',img)
    #     cv2.waitKey(1)
    #     i+=1 
    #     count+=1       
    # #out.release()

    # gif=imageio.mimsave(name,buff,'GIF',duration=0.05)
    # cv2.destroyAllWindows()
