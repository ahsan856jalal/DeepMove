import numpy as np
import cv2
from pylab import *
from os.path import join, isfile
import sys,os,glob
from ctypes import *
import math
import random

video_dataset="Videos" # path to data
gmm_save_dir='gmm_files' # saving directory
temp_dir='temp_dir'
dim = (640, 640) 
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)) # opening kernel
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) # closing kernel
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
videos=os.listdir(video_dataset)
vid_counter=0
for vids in videos:
#    a=input('press enter')
    print('video number {}/93 is in process'.format(vid_counter))
    vid_counter+=1
    if not os.path.exists(join(gmm_save_dir,vids)):
        os.makedirs(join(gmm_save_dir,vids))  
    fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=127,detectShadows=False) # shadow false to avoid large fish contour
    fgbg.setBackgroundRatio(0.7) # set the minimum background ratio
    fgbg.setNMixtures(20) # setting Gaussian distributions
    cap = cv2.VideoCapture(join(video_dataset,vids))
    ret, frame = cap.read()
    frame=cv2.resize(frame,dim,interpolation = cv2.INTER_AREA)
    [img_h,img_w,ch]=shape(frame)
    counter=0
    while(ret):
       frame=cv2.resize(frame,dim,interpolation = cv2.INTER_AREA)
       obj_arr=[]
       blobs=[]
       fgmask = fgbg.apply(frame,) # default settings where learning rate is automaticalyl selected
       fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel1)
       fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel2)
       img_file="%03d.png" % counter
        
       cv2.imwrite(join(gmm_save_dir,vids,img_file),fgmask)
       ret, frame = cap.read()
        
       counter+=1
       k = cv2.waitKey(30) & 0xff
       if k == 27:
           break
    cap.release()


    cv2.destroyAllWindows()