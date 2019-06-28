# Pre-requisites
Python packages: opencv, natsort, math, ctypes, os, glob,sys, numpy, pylab
YOLO darkent installed using guidelines from "https://github.com/AlexeyAB/darknet" or "https://github.com/pjreddie/darknet"
 # GMM Output
Run "opencv_gmm.py" to save gmm frames
# Optical Output
Run "opencv_opencv.py" to save optical frames from the given video
# YOLO Output
After installing yolo from the above mentioned link, make the package and put .cfg,.data,.names and .weights files in the yolo folder and also modify the paths in the net and meta variables of opencv_yolo.py
Run " opencv_yolo.py" to save YOLO detections on the provided video
# Combining Optical and GMM instances
edit the net and meta variable paths before running the file
Run " comb_gmm_optical_images.py" to combing gmm and optical flow instances and do fish detection and classification on all frames
# Preferential Combination
Run "preferential_combination .py" to combine YOLO detections with GMM-Optical combined classified output in a preferential manner where YOLO results will be given preference over the overlapping results and final output will be saved
At the end, a demo will run which will show you original frames vs the processed output frames. It will also give you a table specifying relative occurance of fish species in the given video.
# Visualize the results
Run " show_images.py" by specifying the input folder to see all annotated data from the respective algorithm

##### Step by Step Results ###############
Run files in the following sequence
opencv_gmm.py, opencv_optical.py, opencv_yolo.py, comb_gmm_optical_images.py, preferential_combination.py

###### Fast method ##########
Run these files to get the final result

overall_gmm_optical_classification.py, opencv_yolo.py and at the end preferential_combination.py to compute results
