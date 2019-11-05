import cv2
from os import listdir
from os.path import join, isfile
import xml.etree.ElementTree as ET
import numpy as np
import os


fishclef_train_videos = "Videos" # dir of your video data in this case main dir of the deepsampling
fishclef_train_annos = "Xml"     # similar to the above line
saving_dir="annotated_train_test_comb"  # saving dir for frames and annotations
dim = (640, 640) 
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
def extract_frames(xml_file, video_file,video_name):
    global frame_index, num

    print (xml_file)
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
        print('frame id is {}'.format(frame_id))
        cap.set(1, frame_id)
        ret, img = cap.read()
        if ret:
            img=cv2.resize(img,dim,interpolation = cv2.INTER_AREA)

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
            cv2.imwrite(join(saving_dir,video_name,"%03d.png" % frame_id),img)
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
