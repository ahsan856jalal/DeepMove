# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:34:34 2019

@author: ahsanjalal
"""

import os,sys,glob
import cv2
from pylab import *
font = cv2.FONT_HERSHEY_SIMPLEX
from os.path import join, isfile
from natsort import natsorted, ns


specie_list= ["abudefduf vaigiensis",
             "acanthurus nigrofuscus",
             "amphiprion clarkii",
             "chaetodon lununatus",
             "chaetodon speculum",	
             "chaetodon trifascialis",
             "chromis chrysura",
             "dascyllus aruanus",
             "dascyllus reticulatus",
             "hemigumnus malapterus",
             "myripristis kuntee",
             "neoglyphidodon nigroris",
             "pempheris vanicolensis",
             "plectrogly-phidodon dickii",
            "zebrasoma scopas"]    


comb_gmm_optical='/home/ahsanjalal/Fishclef/MEE_application/comb_gmm_optical_images'
temp_dir='/home/ahsanjalal/Fishclef/MEE_application/temp_dir'
yolo_save_dir='/home/ahsanjalal/Fishclef/MEE_application/yolo_files'
yolo_gmm_optical='/home/ahsanjalal/Fishclef/MEE_application/preferential_comb'
rgb_images='/home/ahsanjalal/Fishclef/MEE_application/gmm_files'
font = cv2.FONT_HERSHEY_SIMPLEX
fish_occurances=np.zeros(shape=(15,1))


text_files=glob.glob(yolo_save_dir+'/'+'*.txt')
text_files=natsorted(text_files)
for text_file in text_files:
    obj_arr=[]
    img=cv2.imread(text_file.split('.txt')[0]+'.png')
    [img_h,img_w,ch]=shape(img)
    filename=text_file.split('/')[-1]
    if filename =='image_135.txt':
        print(filename)
    yolo_text=open(text_file)
    yolo_txt=yolo_text.readlines()
    yolo_text.close()
    comb_text=open(join(comb_gmm_optical,filename))
    gmm_optical_txt=comb_text.readlines()
    comb_text.close()
    if(len(yolo_txt)==0 and len(gmm_optical_txt)!=0):# only gmm_optical detections
        for obj in gmm_optical_txt:
            obj_arr.append(obj)
    if(len(yolo_txt)!=0 and len(gmm_optical_txt)==0): # only yolo detections
        for obj in yolo_txt:
            obj_arr.append(obj)
    if(len(yolo_txt)!=0 and len(gmm_optical_txt)!=0): # chance of overlapping (preferential addition)
        new_optical_gmnm_yolo_txt=[]
        # now we have annotations from yolo as well from gmm optical
        # now the preference is for yolo when overlapping
        for gmm_txt1 in yolo_txt:
                gmm_txt=gmm_txt1.rstrip()
                coords_gmm=gmm_txt.split(' ')
                label_gmm=int(coords_gmm[0])
                w_gmm=round(float(coords_gmm[3])*img_w)
                h_gmm=round(float(coords_gmm[4])*img_h)
                x_gmm=round(float(coords_gmm[1])*img_w)
                y_gmm=round(float(coords_gmm[2])*img_h)
                x_gmm=int(x_gmm)
                y_gmm=int(y_gmm)
                h_gmm=int(h_gmm)
                w_gmm=int(w_gmm)
                xmin_gmm = x_gmm - w_gmm/2
                ymin_gmm = y_gmm - h_gmm/2
                xmax_gmm = x_gmm + w_gmm/2
                ymax_gmm = y_gmm + h_gmm/2  
                if(xmin_gmm<0):
                    xmin_gmm=0
                if(ymin_gmm<0):
                    ymin_gmm=0
                if(xmax_gmm>img_w):
                    xmax_gmm=img_w
                if(ymax_gmm>img_h):
                    ymax_gmm=img_h
                match_flag=0
                count_gt_line=-1
                for line_gt in gmm_optical_txt:
                    count_gt_line+=1
                    line_gt1 = line_gt.rstrip()
                    coords=line_gt1.split(' ')
                    label_gt=int(coords[0])
                
                    w_gt=round(float(coords[3])*img_w)
                    h_gt=round(float(coords[4])*img_h)
                    x_gt=round(float(coords[1])*img_w)
                    y_gt=round(float(coords[2])*img_h)
                    x_gt=int(x_gt)
                    y_gt=int(y_gt)
                    h_gt=int(h_gt)
                    w_gt=int(w_gt)
                    xmin_gt = int(x_gt - w_gt/2)
                    ymin_gt = int(y_gt - h_gt/2)
                    xmax_gt = int(x_gt + w_gt/2)
                    ymax_gt = int(y_gt + h_gt/2)
                    if(xmin_gt<0):
                        xmin_gt=0
                    if(ymin_gt<0):
                        ymin_gt=0
                    if(xmax_gt>img_w):
                        xmax_gt=img_w
                    if(ymax_gt>img_h):
                        ymax_gt=img_h
                # now calculating IOMin 
                
                    xa=max(xmin_gmm,xmin_gt)
                    ya=max(ymin_gmm,ymin_gt)
                    xb=min(xmax_gmm,xmax_gt)
                    yb=min(ymax_gmm,ymax_gt)
                    if(xb>xa and yb>ya):
                        match_flag+=1
                        del gmm_optical_txt[count_gt_line]
                        area_inter=(xb-xa+1)*(yb-ya+1)
                        area_gt=(xmax_gt-xmin_gt+1)*(ymax_gt-ymin_gt+1)
                        area_pred=(xmax_gmm-xmin_gmm+1)*(ymax_gmm-ymin_gmm+1)
                        area_min=min(area_gt,area_pred)
                        area_union=area_pred+area_gt-area_inter
                        if(float(area_inter)/area_union>=0.5):
                            
                            tmp = [int(coords_gmm[0]), float(coords_gmm[1]), float(coords_gmm[2]), float(coords_gmm[3]), float(coords_gmm[4])]
                            new_optical_gmnm_yolo_txt.append(tmp)
                            
                if match_flag==0:
                    # unique yolo output
                                tmp = [int(coords_gmm[0]), float(coords_gmm[1]), float(coords_gmm[2]), float(coords_gmm[3]), float(coords_gmm[4])]
                                new_optical_gmnm_yolo_txt.append(tmp)
        if(len(gmm_optical_txt)!=0):
                for gmm_optical_lines in gmm_optical_txt:
                    gmm_optical_info=gmm_optical_lines.rstrip()
                    coords=gmm_optical_info.split(' ')
                    label_gt=int(coords[0])
                
                    w_gt=round(float(coords[3])*img_w)
                    h_gt=round(float(coords[4])*img_h)
                    x_gt=round(float(coords[1])*img_w)
                    y_gt=round(float(coords[2])*img_h)
                    x_gt=int(x_gt)
                    y_gt=int(y_gt)
                    h_gt=int(h_gt)
                    w_gt=int(w_gt)
                    xmin_gt = int(x_gt - w_gt/2)
                    ymin_gt = int(y_gt - h_gt/2)
                    xmax_gt = int(x_gt + w_gt/2)
                    ymax_gt = int(y_gt + h_gt/2)
                    tmp=[int(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]), float(coords[4])]
                                
                # now the unique gmm_optical remaining
                    new_optical_gmnm_yolo_txt.append(tmp)
    xml_content = ""
    for obj in new_optical_gmnm_yolo_txt:
        xml_content += "%d %f %f %f %f\n" % (obj[0], obj[1], obj[2], obj[3], obj[4])
        w_gt=round(float(obj[3])*img_w)
        h_gt=round(float(obj[4])*img_h)
        x_gt=round(float(obj[1])*img_w)
        y_gt=round(float(obj[2])*img_h)
        x_gt=int(x_gt)
        y_gt=int(y_gt)
        h_gt=int(h_gt)
        w_gt=int(w_gt)
        xmin_gt = int(x_gt - w_gt/2)
        ymin_gt = int(y_gt - h_gt/2)
        xmax_gt = int(x_gt + w_gt/2)
        ymax_gt = int(y_gt + h_gt/2)
        fish_occurances[int(obj[0])]+=1
        cv2.rectangle(img,(xmin_gt,ymin_gt),(xmax_gt,ymax_gt),(255,12,0),2)
        cv2.putText(img,specie_list[int(obj[0])],(x_gt+2+w_gt/2,y_gt+h_gt/2), font, 0.5,(255,0,0),1,cv2.LINE_AA)



    if not os.path.exists(yolo_gmm_optical):
        os.makedirs(yolo_gmm_optical)
    cv2.imwrite(join(yolo_gmm_optical,filename).split('.txt')[0]+'.png',img)
    f = open(join(yolo_gmm_optical,filename), "w")
    f.write(xml_content)
    f.close()

print('Preferential combination is done')

image_files=glob.glob(rgb_images+'/'+'*.png')
image_files=natsorted(image_files)
for img_name in image_files:
    filename=img_name.split('/')[-1]
    original=cv2.imread(join(rgb_images,img_name))
    final_image=cv2.imread(join(yolo_gmm_optical,filename))
    new_image=np.hstack([original,final_image])
    cv2.imshow('original (left) and DeepMove output (right)',new_image)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()
print('Fish classes , Occurance in this video')
print('----------------------------------------')
for x,y in zip(specie_list,fish_occurances):
    print(x,y[0])