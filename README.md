# Pre-requisites
Python packages: opencv, natsort, math, ctypes, os, glob,sys, numpy, pylab
YOLO darkent installed using guidelines from "https://github.com/AlexeyAB/darknet" or "https://github.com/pjreddie/darknet"
 # GMM Output
Run "opencv_gmm.py" to save gmm frames.

In the function "createBackgroundSubtractorMOG2" , you can set different parameters to tune GMM background subtractor for your custom dataset.

Parameter N represents first 'N' frames to be used for background modelling.
We tested it the parameter in the range [20-250].
To set this parameter use "setHistory=int value" 

Another paramter 'NN mixtures' represents total number of mono-gaussian distributions used for training.
We tested it in the range[5-25], increasing the number of gaussians improve results at the stake of computational complexity.
To set it , use "setNMixtures(int nmixtures)"

Similarly, variance threshold for the pixel-model match determines the sensitivity of the model to classify the incoming pixel as foreground or background. We tested this parameter in the range [0.2-0.9]. Increasing the value will make the model less sensitive to small variance. To set this , use "setVarThreshold(double varThreshold)".



# YOLO Output
After installing yolo from the above mentioned link, make the package and put .cfg,.data,.names and .weights files in the yolo folder and also modify the paths in the net and meta variables of opencv_yolo.py
Run " opencv_yolo.py" to save YOLO detections on the provided video
# Combining SOT and GMM instances
edit the net and meta variable paths before running the file
Run " comb_gmm_optical_images.py" to combing gmm and optical flow instances and do fish detection and classification on all frames
# Preferential Combination
Run "preferential_combination .py" to combine YOLO detections with GMM-Optical combined classified output in a preferential manner where YOLO results will be given preference over the overlapping results and final output will be saved
At the end, a demo will run which will show you original frames vs the processed output frames. It will also give you a table specifying relative occurance of fish species in the given video.
# Visualize the results
Run " show_images.py" by specifying the input folder to see all annotated data from the respective algorithm

### Step by Step Results 
Run files in the following sequence
opencv_gmm.py, opencv_optical.py, opencv_yolo.py, comb_gmm_optical_images.py, preferential_combination.py

###### Fast method ##########
Run these files to get the final result

overall_gmm_optical_classification.py, opencv_yolo.py and at the end preferential_combination.py to compute results

# Video Tutorial for the Repo
- You can follow videos in the 'Video_tutorial' folder.
# Steps to reproduce paper results 

1: Clone the directory 

In the main directory, call 

 python create_annotated_data.py  # to make GT frames and annotations from videos and XML files
	
git clone https://github.com/andrewssobral/bgslibrary.git

 cd bgslibrary
 
 pip3 or pip install pybgs
 
  copy making_gmm_detections_bgs_pawcs.py from DeepSampling repo to bgslibrary repo
  
Python or python3 sot.py # will make sot_files

 In the Deepsampling folder:
 
python making_gmm_detections.py # to make GMM files 

### For yolo files

-'git clone https://github.com/pjreddie/darknet.git' will create darknet folder in the home directory.
	  
 goto darknet folder, and edit the Makefile :

first 5 lines are :
GPU=1
CUDNN=1
OPENCV=1
OPENMP=1
DEBUG=0
	
then run 'make -j4 ('4' for quadcore processor otherwise simple make command)' in the darknet folder so that it can make darknet.so file
	  
copy files from '~/DeepSampling->yolo_files' folder into darknet folder and edit the paths in .data files

fish_classification.data:
	line4 --> 'labels = /home/ahsanjalal/darknet_pj/cfg/fishclef.list' according to your path
	
fishclef.data:
	line5 --> 'names = /home/ahsanjalal/darknet_pj/cfg/fishclef.names' according to your path 
	
# Weight files for YOLOv3 and ResNet-50 for fish classification tasks

copy weights from 'https://drive.google.com/open?id=1KvM4-eSDNo5ERrEW6TIeAoNIGxjgPyf6' 

and put them into darknet folder

From DeepSampling folder, run 'python making_yolo_detections_fscore.py' which will save yolo detections in 'yolo_text_files' as text files.

### Combining SOT and GMM

  From DeepSampling directory, run 'python comb_two_bkg_sub_tech.py' after editing paths to net, meta and data folders.
  
  It will combine GMM and SOT in overlap SOT preferential manner, where overlapping blobs with SOT contour information will be classified and saved in 'comb_gmm_sot_overlap_classified' folder.
  
### Preferential Combination of GMM-SOT with YOLO

 Now we will combine temporal algos (GMM-SOT) with feature dependent YOLO outputs in a preferential manner where YOLO output is preferred in overlapping conditions and non-overlapping detections are taken as it is.

Run ' python preferential_combination.py' to get the F-score of DeepSampling classification as given in the paper
	   
