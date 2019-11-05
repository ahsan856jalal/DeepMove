1: Clone the directory 
In the main directory, call 
	* python create_annotated_data.py  # to make GT frames and annotations from videos and XML files
	* git clone https://github.com/andrewssobral/bgslibrary.git
	  cd bgslibrary
	  pip3 or pip install pybgs
	  copy making_gmm_detections_bgs_pawcs.py from DeepSampling repo to bgslibrary repo
	  python or python3 sot.py # will make sot_files
	* In the Deepsampling folder:
	  python making_gmm_detections.py # to make GMM files
	* for yolo files
	  -'git clone https://github.com/pjreddie/darknet.git' will create darknet folder in the home directory.
	  - goto darknet folder, and edit the Makefile :
	  	first 5 lines are :
	  		GPU=1
			CUDNN=1
			OPENCV=1
			OPENMP=1
			DEBUG=0
	  - then run 'make -j4 ('4' for quadcore processor otherwise simple make command)' in the darknet folder so that it can make darknet.so file.

	  -copy files from '~/DeepSampling->yolo_files' folder into darknet folder and edit the paths in .data files
	  		fish_classification.data:
	  			line4 --> 'labels = /home/ahsanjalal/darknet_pj/cfg/fishclef.list' according to your path
	  		fishclef.data:
	  			line5 --> 'names = /home/ahsanjalal/darknet_pj/cfg/fishclef.names' according to your path 

	  - copy weights from 'https://drive.google.com/open?id=1KvM4-eSDNo5ERrEW6TIeAoNIGxjgPyf6' and put them into darknet folder
	  - from DeepSampling folder, run 'python making_yolo_detections_fscore.py' which will save yolo detections in 'yolo_text_files' as text files.

	  # Combining SOT and GMM
	  - From DeepSampling directory, run 'python comb_two_bkg_sub_tech.py' after editing paths to net, meta and data folders.
	  It will combine GMM and SOT in overlap SOT preferential manner, where overlapping blobs with SOT contour information will be classified and saved in 'comb_gmm_sot_overlap_classified' folder.
	  # Preferential Combination of GMM-SOT with YOLO
	  - # Now we will combine temporal algos (GMM-SOT) with feature dependent YOLO outputs in a preferential manner where YOLO output is preferred in overlapping conditions and non-overlapping detections are taken as it is.
	   Run ' python preferential_combination.py' to get the F-score of DeepSampling classification as given in the paper

