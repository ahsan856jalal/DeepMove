######################################

import numpy as np
import cv2
import pybgs as bgs
import sys
import glob
from os.path import join, isfile
import os


video_dataset="/home/muhammadmubeen/Mee_with_tutorial/Videos" # change path according to you data
videos=os.listdir(video_dataset)
sot_save_dir="/home/muhammadmubeen/Mee_with_tutorial/sot_files" # saving dir
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) # morphological opening kernel
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))  # morphological closing kernel
dim = (640, 640) # dimensiion of the data
vid_counter=0

for video_file in videos:
  algorithm=bgs.DPPratiMediod() # SOT method
  print('video number {}/93 is in process'.format(vid_counter))
  vid_counter+=1
  capture = cv2.VideoCapture(join(video_dataset,video_file))

  if not os.path.exists(join(sot_save_dir,video_file)):
        os.makedirs(join(sot_save_dir,video_file))
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
    cv2.imwrite(join(sot_save_dir,video_file,img_file),img_output)
    ret, frame = capture.read()
    counter+=1
    if 0xFF & cv2.waitKey(10) == 27:
        break
      
  # capture.release()
  cv2.destroyAllWindows()

