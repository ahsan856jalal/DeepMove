# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:02:46 2019

@author: ahsanjalal
"""

import numpy as np
import cv2
from pylab import *
from os.path import join, isfile
import sys,os,glob
from ctypes import *
import math
import random
# from natsort import natsorted

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

lib = CDLL("/home/muhammadmubeen/darknet_pj/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.2, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res
    
net = load_net("/home/muhammadmubeen/darknet_pj/resnet50.cfg", "/home/muhammadmubeen/darknet_pj/resnet50_300.weights", 0)
meta = load_meta("/home/muhammadmubeen/darknet_pj/fish_classification.data") # change paths in net and meta 


dim = (640, 640) 
gmm_save_dir='gmm_files' # gmm Directory
sot_save_dir='sot_files' # SOT directory
comb_gmm_sot='comb_gmm_sot_overlap_classified' # saving dir
temp_dir='temp_dir'
rgb_images='annotated_train_test_comb'
rgb_dir='/home/muhammadmubeen/Mee_with_tutorial/annotated_train_test_comb' # RGB frames path


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

val_list=open('val_sort.txt') #test file path present in DeepSampling
val_lines=val_list.readlines()
count=0
[img_h,img_w,ch]=[640,640,3]
if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
for img_name in val_lines:
    img_name=join(rgb_dir,img_name)
    print('count is {}'.format(count))
    count+=1    
    blobs=[]
    obj_arr=[]
    img_name = img_name.rstrip()
    filename=img_name.split('/')
    img_file=filename[-1]
    video_file=filename[-2]
    if not os.path.exists(join(gmm_save_dir,video_file,img_file)):
        gmm_img=np.zeros(shape=(img_h,img_w))
        print('gmm file not present')
    else:
        gmm_img=cv2.imread(join(gmm_save_dir,video_file,img_file))
        gmm_img=cv2.resize(gmm_img,dim,interpolation = cv2.INTER_AREA)
    filename=img_name.split('/')[-1]
    rgb_img=cv2.imread(join(rgb_images,video_file,img_file),0)
    if not os.path.exists(join(sot_save_dir,video_file,img_file)):
        sot_img=np.zeros(shape=(img_h,img_w))
        print('sot file not present')
    else:
        sot_img=cv2.imread(join(sot_save_dir,video_file,img_file),0)
        sot_img=cv2.resize(sot_img,dim,interpolation = cv2.INTER_AREA)# it has different resolution so resize to 640 square is a must !!
    # Now we have multisot and GMM frames
    if len(shape(gmm_img))>2:
        gmm_img=cv2.cvtColor(gmm_img, cv2.COLOR_RGB2GRAY)
    ret,threshed_img_gmm= cv2.threshold(gmm_img,55, 255, cv2.THRESH_BINARY)
    if( np.max(threshed_img_gmm)>0):
        _, contours_gmm, hier= cv2.findContours(threshed_img_gmm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_poly_gmm = [None]*len(contours_gmm)
    else:
        contours_gmm=[]
    boundRect_gmm = np.array([[0,0,0,0]]) #first vague entry
    if contours_gmm:
            for i, c in enumerate(contours_gmm):
                if(cv2.contourArea(c)>300 and cv2.contourArea(c)<85000):
                    contours_poly_gmm[i] = cv2.approxPolyDP(c, 3, True)
                    boundRect_gmm = vstack([boundRect_gmm,cv2.boundingRect(contours_poly_gmm[i])])
                    
                # now we have bounding boxes of all
    if shape(boundRect_gmm)[0]>1:
        boundRect_gmm=boundRect_gmm[1:,:]
    else:
        boundRect_gmm=[]
    
    
    #sot
    if len(shape(sot_img))>2:
        sot_img=cv2.cvtColor(sot_img, cv2.COLOR_RGB2GRAY)
    ret,threshed_img_sot= cv2.threshold(sot_img,55, 255, cv2.THRESH_BINARY)
    if( np.max(threshed_img_sot)>0):
        _, contours_sot, hier= cv2.findContours(threshed_img_sot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_poly_sot = [None]*len(contours_sot)
    else:
        contours_sot=[]
    boundRect_sot = np.array([[0,0,0,0]]) #first vague entry
    if contours_sot:
            for i, c in enumerate(contours_sot):
                if(cv2.contourArea(c)>300 and cv2.contourArea(c)<85000):
                    contours_poly_sot[i] = cv2.approxPolyDP(c, 3, True)
                    boundRect_sot = vstack([boundRect_sot,cv2.boundingRect(contours_poly_sot[i])])
                    
                # now we have bounding boxes of all
    if shape(boundRect_sot)[0]>1:
        boundRect_sot=boundRect_sot[1:,:]
    else:
        boundRect_sot=[]
    
    # now select only overlapping
    obj_arr=[]
    new_img=np.zeros(shape=(img_h,img_w))
    for x1,y1,w1,h1 in boundRect_gmm:
        xmin_gmm=x1
        xmax_gmm=x1+w1
        ymin_gmm=y1
        ymax_gmm=y1+h1
        for x2,y2,w2,h2 in boundRect_sot:
            xmin_sot=x2
            xmax_sot=x2+w2
            ymin_sot=y2
            ymax_sot=y2+h2
            xa=max(xmin_gmm,xmin_sot)
            ya=max(ymin_gmm,ymin_sot)
            xb=min(xmax_gmm,xmax_sot)
            yb=min(ymax_gmm,ymax_sot)
            if(xb>xa and yb>ya):
                area_inter=(xb-xa+1)*(yb-ya+1)
                area_gt=(xmax_gmm-xmin_gmm+1)*(ymax_gmm-ymin_gmm+1)
                area_pred=(xmax_sot-xmin_sot+1)*(ymax_sot-ymin_sot+1)
                area_min=min(area_gt,area_pred)
                area_union=area_pred+area_gt-area_inter
                if(float(area_inter)/area_min>=0.5):
                    img_patch=threshed_img_sot[ymin_sot:ymax_sot,xmin_sot:xmax_sot]
                    img_patch = cv2.resize(img_patch.astype('float32'), dsize=(50,50))
                    cv2.imwrite(temp_dir+'/'+ "true_image.png", img_patch)
                    im = load_image(temp_dir+'/'+ "true_image.png", 0, 0)
                    r = classify(net, meta, im)
                    r=r[0]
                    fish_label_det=specie_list.index(r[0])
                    if(fish_label_det!=15 and r[1]>0.8):# previous threshold set was o.4 but FP was very high 4362

                        x=xmin_sot
                        y=ymin_sot
                        w=xmax_sot-xmin_sot
                        h=ymax_sot-ymin_sot
                        x = (ymin_sot+w/2.0) / img_w
                        y = (y+h/2.0) / img_h
                        w = float(w) / img_w
                        h = float(h) / img_h
                        fish_specie=fish_label_det
                        tmp = [fish_specie, x, y, w, h]
                        obj_arr.append(tmp)
                        new_img[ymin_sot:ymax_sot,xmin_sot:xmax_sot]=threshed_img_sot[ymin_sot:ymax_sot,xmin_sot:xmax_sot]
                         

                            
        
    xml_content = ""
    for obj in obj_arr:
        xml_content += "%d %f %f %f %f\n" % (obj[0], obj[1], obj[2], obj[3], obj[4])
    if not os.path.exists(join(comb_gmm_sot,video_file)):
        os.makedirs(join(comb_gmm_sot,video_file))
    f = open(join(comb_gmm_sot,video_file,filename).split('.png')[0]+'.txt', "w")
    f.write(xml_content)
    f.close()
    cv2.imwrite(join(comb_gmm_sot,video_file,filename),new_img)
            
        
        