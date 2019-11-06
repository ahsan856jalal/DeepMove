######################################

import numpy as np
import cv2
import pybgs as bgs
import sys
import glob
from os.path import join, isfile
import os

vid_name='/home/muhammadmubeen/MEE_application/9c333821ab0e2a9e4c5209065c415309#201102031130_s3_0.flv'
sot_save_dir="/home/muhammadmubeen/MEE_application/sot_file_for_video" # saving dir
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) # morphological opening kernel
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))  # morphological closing kernel
dim = (640, 640) # dimensiion of the data
vid_counter=0

algorithm=bgs.DPPratiMediod() # SOT method, to change its parameter you need to change its xml file in bgslibrary folder 
print('video number {} is in process'.format(vid_name))
vid_counter+=1
capture = cv2.VideoCapture(vid_name)

if not os.path.exists(sot_save_dir):
      os.makedirs(sot_save_dir)
ret, frame = capture.read()
frame=cv2.resize(frame,dim,interpolation = cv2.INTER_AREA)
counter=0
while ret:
  frame=cv2.resize(frame,dim,interpolation = cv2.INTER_AREA)
  pos_frame = capture.get(1)
  img_output = algorithm.apply(frame)
  img_output = cv2.morphologyEx(img_output, cv2.MORPH_OPEN, kernel1)
  img_output = cv2.morphologyEx(img_output, cv2.MORPH_CLOSE, kernel2)
  img_file="%03d.png" % counter
  cv2.imwrite(join(sot_save_dir,img_file),img_output)
  ret, frame = capture.read()
  counter+=1
  if 0xFF & cv2.waitKey(10) == 27:
      break
    
# capture.release()
cv2.destroyAllWindows()

