# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:35:59 2018

@author: ahsanjalal
"""



import sys,os,glob
import numpy as np
from os.path import join, isfile
from sklearn import *
from pylab import *
from PIL import Image
import cv2
import dlib
from scipy.misc import imresize
from statistics import mode
from tempfile import TemporaryFile
from collections import Counter
import numpy
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
#from rgb2gray import rgb2gray
import lxml.etree
import scipy.misc
from natsort import natsorted, ns
import xml.etree.ElementTree as ET
from shutil import copytree
import matplotlib.pyplot as plt


saving_test_dir='/home/ahsanjalal/Fishclef/Datasets/Test_dataset/val_sort_gmm_optical_mixed'
#original RGB images
#gt_train='/home/ahsanjalal/Fishclef/Datasets/Test_dataset/annotated_frames_train'
#gt_test='/home/ahsanjalal/Fishclef/Datasets/Test_dataset/annotated_frames'
# Gmm frames
#gmm_test='/home/ahsanjalal/Fishclef/Datasets/Test_dataset/video_gmm_results_bkgRatio_07_numframe_250_ga_20_sz_200_disk'
#gmm_train='/home/ahsanjalal/Fishclef/Datasets/Test_dataset/Train_video_gmm_results_bkgRatio_07_numframe_250_ga_20_sz_200_disk'
## optical frames
#optical_test='/home/ahsanjalal/Optical_flow'
#optical_train='/home/ahsanjalal/Train_Optical_flow'

optical_comb='/home/ahsanjalal/optical_train_test_comb'
gmm_comb='/home/ahsanjalal/Fishclef/Datasets/Test_dataset/gmm_train_test_comb'
gt_comb='/home/ahsanjalal/Fishclef/Datasets/Test_dataset/annotated_train_test_comb'


val_sort=open('/home/ahsanjalal/Fishclef/Datasets/Test_dataset/val_sort.txt')
val_test=val_sort.readlines()


test_tmp=0
img_height,img_width=[640,640]
for img_name1 in val_test:
    test_tmp+=1
    print(test_tmp)
    img_name = img_name1.rstrip()
    video_file=img_name.split('/')[-2]
    img_file=img_name.split('/')[-1]
    
    # Now reading RGB,GMM,OPTICAL and YOLO output
    img_rgb=cv2.imread(join(gt_comb,video_file,img_file))
    img_gt=np.array(img_rgb)
    if os.path.exists(join(gmm_comb,video_file,img_file)):
        img_gmm=cv2.imread(join(gmm_comb,video_file,img_file))
    else:
        img_gmm=np.zeros(shape=[640,640,3])
    if os.path.exists(join(optical_comb,video_file,img_file)):
        img_optical=cv2.imread(join(optical_comb,video_file,img_file))
        img_optical=imresize(img_optical,[640,640])# it has different resolution so resize to 640 square is a must !!
    else:
        img_optical=np.zeros(shape=[640,640,3])
#    img_yolo=cv2.imread(join(yolo_test,video_file,img_file))
    
    img_gt[:,:,0]=0  # no gray channel             
#            img_gt[:,:,0]=img_gt_gray
    img_gt[:,:,1]=img_gmm[:,:,0]
    img_gt[:,:,2]=img_optical[:,:,0]
#    img_gmm_optical_gray=cv2.cvtColor(img_gt,cv2.COLOR_BGR2GRAY)
#    img_yolo=cv2.cvtColor(img_yolo,cv2.COLOR_BGR2GRAY)
#    img_gt[:,:,0]=0  # no gray channel             
#    img_gt[:,:,1]=img_gmm_optical_gray
#    img_gt[:,:,2]=img_yolo
    ret, threshed_img = cv2.threshold(cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY),
                20, 255, cv2.THRESH_BINARY)
    image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    obj_arr = []
    if contours:
        for c in range(len(contours)):
            if(hier[0,c,3]==-1):
            # get the bounding rect
                if(cv2.contourArea(contours[c])>=600 and cv2.contourArea(contours[c])<120000):
                    x, y, w, h = cv2.boundingRect(contours[c])
#                    cv2.rectangle(img_gt, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    x = (x+w/2.0) / img_width
                    y = (y+h/2.0) / img_height
                    w = float(w) / img_width
                    h = float(h) / img_height
                    fish_specie=0
                    tmp = [fish_specie, x, y, w, h]
                    obj_arr.append(tmp)
    xml_content = ""
    for obj in obj_arr:
        xml_content += "%d %f %f %f %f\n" % (obj[0], obj[1], obj[2], obj[3], obj[4])
    if not os.path.exists(join(saving_test_dir,video_file)):
        os.makedirs(join(saving_test_dir,video_file))
    f = open(join(saving_test_dir,video_file,img_file).split('.png')[0]+'.txt', "w")
    f.write(xml_content)
    f.close()
    cv2.imwrite(join(saving_test_dir,video_file,img_file),img_rgb)
    


