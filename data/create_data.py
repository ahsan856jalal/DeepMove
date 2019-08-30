# This script creates the FishCLEF2015 dataset to be used by caffe.
# It converts the data into VOC convention and later the same VOC scripts are used to generate the dataset
import cv2
from os import listdir
from os.path import join, isfile
import xml.etree.ElementTree as ET
import numpy as np
import os
from pylab import *
from scipy.misc import imresize
import scipy.misc

fishclef_train_videos = "Videos"
# fishclef_test_videos = "/home/ahsanjalal/Fishclef/Datasets/Test_dataset/Videos"

fishclef_train_annos = "Xml"
# fishclef_test_annos = "/home/ahsanjalal/Fishclef/Datasets/Test_dataset/Ground Truth XML"
saving_dir="annotated_train_test_comb"
#frame_out_dir = "/home/ahsanjalal/Fishclef/Datasets/Test_dataset/JPEGImages"
#anno_out_dir = "/home/ahsanjalal/Fishclef/Datasets/Test_dataset/JPEGImages"
#if not os.path.exists(frame_out_dir):
#    os.makedirs(frame_out_dir)
#if not os.path.exists(anno_out_dir):
#    os.makedirs(anno_out_dir)
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)


fish_list = ["abudefduf vaigiensis",
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

num = np.zeros(15)
# num=10000*num
def extract_frames(xml_file, video_file,video_name):
    global frame_index, num

    print xml_file
    # xml_file is the FishCLEF2015 annotations file
    # video_file is the video
    # frame_index is the starting frame index number
    tree = ET.parse(xml_file)
    root = tree.getroot()
    if not os.path.exists(join(saving_dir,video_name)):
        os.makedirs(join(saving_dir,video_name))
    cap = cv2.VideoCapture(video_file)
    vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    
    for frame in root.findall("frame"):
        frame_id = int(frame.get("id"))
        cap.set(1, frame_id)
        ret, img = cap.read()
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        img=imresize(img,[640,640])
        # cv2.imwrite(join(saving_dir,video_name, "%03d.png" % frame_id), img)
        #cap.release()

        obj_arr = []
        for obj in frame.findall("object"):
            fish_specie = obj.get("fish_species")
            if not fish_specie:
                fish_specie = obj.get("species_name")
            fish_specie = fish_specie.lower()
            if fish_specie == "chaetodon lunulatus":
                fish_specie = "chaetodon lununatus"

            # check on the fish specie
            if fish_specie == "null":
                continue
            if fish_specie not in fish_list:
                print "ERROR: %s" % fish_specie
                continue

            
            h = int(obj.get("h"))
            w = int(obj.get("w"))
            x = int(obj.get("x"))
            y = int(obj.get("y"))

            x = float(x+w/2.0) / vid_width
            y = float(y+h/2.0) / vid_height
            w = float(w) / vid_width
            h = float(h) / vid_height

            fish_specie = fish_list.index(fish_specie)
            tmp = [fish_specie, x, y, w, h]
            obj_arr.append(tmp)
            num[fish_specie] += 1
        f = open(join(saving_dir,video_name, "%03d.txt" % frame_id), "w")
        f.write(generate_xml(obj_arr, frame_index, vid_width, vid_height));
        f.close()
        scipy.misc.imsave(join(saving_dir,video_name,"%03d.png" % frame_id),img)
        # cap.release()
        frame_index += 1

def generate_xml(objects, frame_index, width, height):
    xml_content = ""
    
    for obj in objects:
        xml_content += "%d %f %f %f %f\n" % (obj[0], obj[1], obj[2], obj[3], obj[4])

    return xml_content

frame_index = 0

## Extracting the trainval list
trainval_annos = [f for f in listdir(fishclef_train_annos) if isfile(join(fishclef_train_annos, f))]

for trainval_anno in trainval_annos:
   video_name="%s.flv" % trainval_anno[:-4]
   video_file = join(fishclef_train_videos, "%s.flv" % trainval_anno[:-4])
   xml_file = join(fishclef_train_annos, trainval_anno)

   extract_frames(xml_file, video_file,video_name)

trainval_num = frame_index

##Extracting the test list
# test_annos = [f for f in listdir(fishclef_test_annos) if isfile(join(fishclef_test_annos, f))]

# for test_anno in test_annos:
#     video_name="%s.flv" % test_anno[:-4]
#     if not os.path.exists(saving_dir+video_name):
#         os.makedirs(saving_dir+video_name)
#     video_file = join(fishclef_test_videos, "%s.flv" % test_anno[:-4])
#     xml_file = join(fishclef_test_annos, test_anno)

#     extract_frames(xml_file, video_file,video_name)

# test_num = frame_index

# Writing the list to trainval.txt and test.txt
#f = open("trainval.txt", "w")
#for i in range(0, trainval_num):
#    f.write("%08d\n" % i)
#f.close()

#f = open("test.txt", "w")
#for i in range(trainval_num, test_num):
#    f.write("%08d\n" % i)
#f.close()
for i in range(0, 15):
    print fish_list[i], num[i]
print('sum of all fish are {}'.format(sum(num)))