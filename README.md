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


# Optical Output
Run "opencv_opencv.py" to save optical frames from the given video.
In this file , you can set different parameters of the Optical flow's algorithm using function "calcOpticalFlowFarneback".
Parameter "pyr_scale" specifying the image scale (<1) to build pyramids for each image; pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one. 
Parameter "levels" 	specifies the number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images are used. We tested this in the range [1-10].
Parameter "winSize" represents averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field. We tried this in the range [3-21].
Paramter "iterations" 	represents the number of iterations the algorithm does at each pyramid level. We tested this parameter in the range [1-5] as it is computational expensive.
Paramter "poly_n" represents the size of the pixel neighborhood used to find polynomial expansion in each pixel; larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7. 
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
