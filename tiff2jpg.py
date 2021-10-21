from cv2 import cv2 
import os
import shutil
import numpy as np


# tif_path=r'raw_data\01_raw\test2_00003.tif'#r'raw_data\01_raw\back_00001.tif'
# save_path='test1.jpg'

# img=cv2.imread(tif_path,0)
# img=img/np.max(img)*255
# img=img.astype(np.uint8)

# cv2.imwrite(save_path,img)

# cv2.imshow('p',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cap=cv2.VideoCapture('PIVlab_out.avi')

count=0
while True:
    r,f=cap.read()
    if f is None:
        break
    cv2.imwrite("{0:05d}.jpg".format(count),f)
    count+=1
    cv2.imshow('p',f)
    cv2.waitKey(0)
    

cap.release()
cv2.destroyAllWindows()