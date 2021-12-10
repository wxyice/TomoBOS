'''
光流法计算脚本

可以使用光流算法替代互相关，分辨率可以得到提高

'''

from cv2 import cv2
import cv2
import matplotlib.pyplot as plt  # 直接用cv2中的imshow也可以画图，这里用plt绘图方便显示
import numpy as np



def Defection_sencing(img1,img2,calibration=9e-5):
    flow=cv2.calcOpticalFlowFarneback(img1, img2,flow=None, pyr_scale=0.5, levels=1, winsize=5, iterations=30, poly_n=5, poly_sigma=1.2, flags=0)
    flow=flow*9e-5
    return flow


path1=r'raw2jpg\back.jpg'
path2=r'raw2jpg\test1.jpg'


old_img=cv2.imread(path1,0)


mask = np.zeros_like(old_img)
x=np.zeros_like(old_img)
y=np.zeros_like(old_img)

new_img=cv2.imread(path2,0)

flow=Defection_sencing(old_img,new_img)



fig,ax=plt.subplots(1,2)

ax[0].imshow(flow[...,0])
ax[1].imshow(flow[...,1])

plt.savefig('OF-winsize=5-iter=30.jpg')
plt.show()




# k = cv2.waitKey(0) & 0xff
# cv2.destroyAllWindows()
