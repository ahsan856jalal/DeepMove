# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:07:52 2019

@author: ahsanjalal
"""
import os,sys,glob
import cv2
from pylab import *
font = cv2.FONT_HERSHEY_SIMPLEX
from natsort import natsorted, ns

main_dir='/home/ahsanjalal/Fishclef/MEE_application/preferential_comb/'
os.chdir(main_dir)
specie_list= ["vaigiensis",
             "nigrofuscus",
             "clarkii",
             "lununatus",
             "speculum",    
             "trifascialis",
             "chrysura",
             "aruanus",
             "reticulatus",
             "malapterus",
             "kuntee",
             "nigroris",
             "vanicolensis",
             "dickii",
            "scopas",
            "background"]   
##### show selective image######
#
#name='image_267'      
#img=cv2.imread(name+'.png')
#height,width,ch=shape(img)
#a=open(name+'.txt')
#text=a.readlines()
#for line in text:
#    line = line.rstrip()
#    coords=line.split(' ')
#    w=float(coords[3])*width
#    h=float(coords[4])*height
#    x=float(coords[1])*width
#    y=float(coords[2])*height
#    x=int(x)
#    y=int(y)
#    h=int(h)
#    w=int(w)
#    xmin = x - w/2
#    ymin = y - h/2
#    xmax = x + w/2
#    ymax = y + h/2
#    img=cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,12,0),2)
#    cv2.putText(img,specie_list[int(coords[0])],(x+2+w/2,y+h/2), font, 0.5,(255,0,0),1,cv2.LINE_AA)
#    
#imshow(img)
#

####################  showing directory images
image_files=glob.glob(main_dir+'*.png')
image_files=natsorted(image_files)
for img_name in image_files:
    img=cv2.imread(img_name)
    height,width,ch=shape(img)
    a=open(img_name.split('.png')[0]+'.txt')
    text=a.readlines()
    for line in text:
        line = line.rstrip()
        coords=line.split(' ')
        w=float(coords[3])*width
        h=float(coords[4])*height
        x=float(coords[1])*width
        y=float(coords[2])*height
        x=int(x)
        y=int(y)
        h=int(h)
        w=int(w)
        xmin = x - w/2
        ymin = y - h/2
        xmax = x + w/2
        ymax = y + h/2
        img=cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,12,0),2)
        cv2.putText(img,specie_list[int(coords[0])],(x+2+w/2,y+h/2), font, 0.5,(255,0,0),1,cv2.LINE_AA)
    
    cv2.imshow('asa',img)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()


    