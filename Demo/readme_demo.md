# steps to make files
 Run :
 
 1) gmm_for_video.py to get rgb and gmm frames
 
 2) yolo_for_video.py to get yolo frames
 
 3) Copy 'sot_for_video.py' into ~/bgslibrary folder and run
 'python3 sot_for_video.py' to get SOT frames saved in ~/Demo folder
 
 4) Go back to Demo fodler and run
 'comb_gmm_sot_video.py' to get gmm_sot classified output
 
 5) Run 'preferential_combination_for_video.py' to save final output in 'prefertial_comb' folder also see the comparison betwen original and 
 final output in the form of a GUI.
